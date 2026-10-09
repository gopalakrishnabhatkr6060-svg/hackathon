import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def resolve_file(filename):
    for p in [Path(__file__).resolve().parent / filename, Path(__file__).resolve().parent.parent / filename]:
        if p.exists():
            return p
    return Path(__file__).resolve().parent / filename

MODEL_PATH = resolve_file("xgb_supplier_risk_model.json")
DATASET_PATH = resolve_file("suppliers_dataset.csv")
PDF_KNOWLEDGE_PATH = Path(__file__).resolve().parent / "db" / "pdf_knowledge.json"

FEATURES = [
    "avg_price_creep",
    "rec_rejection_pct",
    "rejection_spike_ratio",
    "min_days_to_expiry",
    "criticality_numeric",
    "stock_cover_days",
    "is_single_source"
]

def get_model():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model

def predict_single(data):
    model = get_model()
    row = {f: float(data.get(f, 0.0)) for f in FEATURES}
    df = pd.DataFrame([row], columns=FEATURES)
    
    pred = int(model.predict(df)[0])
    
    try:
        probs = model.predict_proba(df)[0]
        risk_prob = float(probs[1])
    except Exception:
        risk_prob = 1.0 if pred == 1 else 0.0
        
    return {
        "status": "success",
        "prediction": pred,
        "is_high_risk": pred == 1,
        "risk_probability": round(risk_prob * 100, 1),
        "input": row
    }

def batch_predict(records):
    model = get_model()
    rows = []
    for r in records:
        row = {f: float(r.get(f, 0.0)) for f in FEATURES}
        rows.append(row)
        
    df = pd.DataFrame(rows, columns=FEATURES)
    preds = model.predict(df)
    
    try:
        probs = model.predict_proba(df)[:, 1]
    except Exception:
        probs = [1.0 if p == 1 else 0.0 for p in preds]
        
    results = []
    for i, orig in enumerate(records):
        item = dict(orig)
        p = int(preds[i])
        pr = round(float(probs[i]) * 100, 1)
        item["risk_pred"] = p
        item["is_high_risk"] = p == 1
        item["risk_probability"] = pr
        item["status_label"] = "🚨 High Risk" if p == 1 else "✅ Stable"
        results.append(item)
        
    high_count = sum(1 for r in results if r["is_high_risk"])
    
    return {
        "status": "success",
        "total_evaluated": len(results),
        "high_risk_count": high_count,
        "stable_count": len(results) - high_count,
        "risk_rate_pct": round((high_count / max(len(results), 1)) * 100, 1),
        "records": results
    }

def get_feature_importance():
    model = get_model()
    if hasattr(model, "feature_importances_"):
        imps = [float(x) for x in model.feature_importances_]
        total = sum(imps) or 1.0
        imp_dict = [
            {"feature": f, "importance": round(imp, 4), "percentage": round((imp / total) * 100, 1)}
            for f, imp in zip(FEATURES, imps)
        ]
        imp_dict.sort(key=lambda x: x["importance"], reverse=True)
        return {"status": "success", "features": imp_dict}
    return {"status": "error", "message": "No feature importances found"}

def what_if_sensitivity(base_data, feature_name, steps=10):
    model = get_model()
    base_row = {f: float(base_data.get(f, 0.0)) for f in FEATURES}
    
    ranges = {
        "avg_price_creep": (-0.1, 1.2),
        "rec_rejection_pct": (0.0, 0.5),
        "rejection_spike_ratio": (0.5, 6.0),
        "min_days_to_expiry": (10, 360),
        "criticality_numeric": (1.0, 5.0),
        "stock_cover_days": (2.0, 90.0),
        "is_single_source": (0, 1)
    }
    
    r_min, r_max = ranges.get(feature_name, (0.0, 1.0))
    test_vals = np.linspace(r_min, r_max, steps)
    
    curve = []
    for val in test_vals:
        sim_row = dict(base_row)
        sim_row[feature_name] = float(val)
        sim_df = pd.DataFrame([sim_row], columns=FEATURES)
        try:
            prob = float(model.predict_proba(sim_df)[0][1]) * 100
        except Exception:
            prob = 100.0 if model.predict(sim_df)[0] == 1 else 0.0
            
        curve.append({
            "val": round(float(val), 3),
            "risk_prob": round(prob, 1)
        })
        
    return {
        "status": "success",
        "feature": feature_name,
        "curve": curve
    }

def rag_query(query, top_k=4):
    docs = []
    metadata = []
    
    # 1. Ingest Indian Suppliers Data
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
        for _, r in df.iterrows():
            ss_text = "Single-source sole provider (High dependency)" if r.get('is_single_source', 0) == 1 else "Multi-sourced (Backup vendors available)"
            doc = (
                f"Supplier ID: {r.get('supplier_id', 'N/A')} | Name: {r.get('supplier_name', 'N/A')} | "
                f"Country: India | Category: {r.get('category', 'N/A')} | Hub/Region: {r.get('region', 'N/A')} | "
                f"Spend Tier: {r.get('spend_tier', 'N/A')} | Price Creep Ratio: {r.get('avg_price_creep', 0):.4f} | "
                f"Recent Rejection Rate: {r.get('rec_rejection_pct', 0)*100:.1f}% | "
                f"Rejection Spike Ratio: {r.get('rejection_spike_ratio', 1.0):.2f}x | "
                f"Stock Cover: {r.get('stock_cover_days', 0):.1f} days runway | "
                f"Days to Expiry: {r.get('min_days_to_expiry', 0)} days | "
                f"Component Criticality: Tier {int(r.get('criticality_numeric', 1))} | "
                f"Sourcing Redundancy: {ss_text}."
            )
            docs.append(doc)
            
            metadata.append({
                "type": "supplier",
                "id": r.get('supplier_id'),
                "name": r.get('supplier_name'),
                "category": r.get('category'),
                "region": r.get('region'),
                "price_creep": float(r.get('avg_price_creep', 0)),
                "rejection_pct": float(r.get('rec_rejection_pct', 0)),
                "stock_cover": float(r.get('stock_cover_days', 0)),
                "is_single_source": int(r.get('is_single_source', 0)),
                "target": int(r.get('target', 0))
            })
            
    # 2. Ingest Uploaded PDF Document Chunks
    if PDF_KNOWLEDGE_PATH.exists():
        try:
            with open(PDF_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                pdf_data = json.load(f)
                for item in pdf_data:
                    doc = (
                        f"[DOCUMENT: {item.get('filename')}] Chunk {item.get('chunk_id')}: "
                        f"{item.get('text')}"
                    )
                    docs.append(doc)
                    metadata.append({
                        "type": "pdf_document",
                        "filename": item.get('filename'),
                        "chunk_id": item.get('chunk_id'),
                        "title": item.get('title', item.get('filename')),
                        "upload_time": item.get('upload_time')
                    })
        except Exception:
            pass
            
    if not docs:
        return {"status": "error", "message": "Knowledge base is empty"}
        
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform(docs)
    q_vec = vec.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    
    top_indices = sims.argsort()[-top_k:][::-1]
    results = []
    
    model = get_model()
    
    for idx in top_indices:
        sim_score = round(float(sims[idx]), 3)
        if sim_score <= 0.005 and len(results) > 0:
            continue
            
        meta = metadata[idx].copy()
        meta["score"] = sim_score
        meta["doc_text"] = docs[idx]
        
        if meta["type"] == "supplier":
            pred_df = pd.DataFrame([{
                "avg_price_creep": meta["price_creep"],
                "rec_rejection_pct": meta["rejection_pct"],
                "rejection_spike_ratio": 1.0,
                "min_days_to_expiry": 180,
                "criticality_numeric": 2.0,
                "stock_cover_days": meta["stock_cover"],
                "is_single_source": meta["is_single_source"]
            }], columns=FEATURES)
            try:
                meta["risk_pred"] = int(model.predict(pred_df)[0])
            except Exception:
                meta["risk_pred"] = meta.get("target", 0)
        else:
            meta["risk_pred"] = -1
            
        results.append(meta)
        
    return {
        "status": "success",
        "query": query,
        "total_knowledge_docs": len(docs),
        "results": results
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "No command provided"}))
        sys.exit(1)
        
    cmd = sys.argv[1]
    
    try:
        if cmd == "predict":
            input_json = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
            data = json.loads(input_json)
            res = predict_single(data)
            print(json.dumps(res))
            
        elif cmd == "batch":
            input_json = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
            data = json.loads(input_json)
            res = batch_predict(data)
            print(json.dumps(res))
            
        elif cmd == "importance":
            res = get_feature_importance()
            print(json.dumps(res))
            
        elif cmd == "sensitivity":
            feature_name = sys.argv[2]
            base_json = sys.argv[3] if len(sys.argv) > 3 else "{}"
            base_data = json.loads(base_json)
            res = what_if_sensitivity(base_data, feature_name)
            print(json.dumps(res))
            
        elif cmd == "rag":
            query = sys.argv[2]
            top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
            res = rag_query(query, top_k)
            print(json.dumps(res))
            
        else:
            print(json.dumps({"status": "error", "message": f"Unknown command: {cmd}"}))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
