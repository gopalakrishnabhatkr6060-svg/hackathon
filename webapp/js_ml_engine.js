const fs = require('fs');
const path = require('path');

const FEATURES = [
    "avg_price_creep",
    "rec_rejection_pct",
    "rejection_spike_ratio",
    "min_days_to_expiry",
    "criticality_numeric",
    "stock_cover_days",
    "is_single_source"
];

function resolveFile(filename) {
    const localPath = path.join(__dirname, filename);
    if (fs.existsSync(localPath)) return localPath;
    const parentPath = path.join(__dirname, '..', filename);
    if (fs.existsSync(parentPath)) return parentPath;
    return localPath;
}

let cachedModel = null;
function getModelData() {
    if (!cachedModel) {
        const modelPath = resolveFile('xgb_supplier_risk_model.json');
        if (fs.existsSync(modelPath)) {
            cachedModel = JSON.parse(fs.readFileSync(modelPath, 'utf-8'));
        }
    }
    return cachedModel;
}

function evaluateBooster(input) {
    const model = getModelData();
    if (!model || !model.learner || !model.learner.gradient_booster) {
        // Fallback rule-based estimation if model file missing
        const creep = Number(input.avg_price_creep || 0);
        const rej = Number(input.rec_rejection_pct || 0);
        const spike = Number(input.rejection_spike_ratio || 1);
        const prob = (creep > 0.3655 || (rej > 0.15 && spike > 1.8)) ? 0.95 : 0.05;
        return { prob, pred: prob >= 0.5 ? 1 : 0 };
    }

    const featureNames = model.learner.feature_names || FEATURES;
    const trees = model.learner.gradient_booster.model.trees || [];

    let margin = 0;
    for (const tree of trees) {
        let node = 0;
        while (tree.left_children && tree.left_children[node] !== -1 && tree.left_children[node] !== undefined) {
            const featIdx = tree.split_indices[node];
            const featName = featureNames[featIdx] || FEATURES[featIdx];
            const val = Number(input[featName] !== undefined ? input[featName] : 0);
            const cond = tree.split_conditions[node];
            if (val < cond) {
                node = tree.left_children[node];
            } else {
                node = tree.right_children[node];
            }
        }
        if (tree.split_conditions && tree.split_conditions[node] !== undefined) {
            margin += tree.split_conditions[node];
        }
    }

    const prob = 1 / (1 + Math.exp(-margin));
    return { prob, pred: prob >= 0.5 ? 1 : 0 };
}

function predictSingle(data) {
    const row = {};
    for (const f of FEATURES) {
        row[f] = Number(data[f] !== undefined ? data[f] : 0);
    }
    const { prob, pred } = evaluateBooster(row);
    return {
        status: "success",
        prediction: pred,
        is_high_risk: pred === 1,
        risk_probability: Number((prob * 100).toFixed(1)),
        input: row
    };
}

function batchPredict(records) {
    const results = [];
    let highCount = 0;

    for (const orig of records) {
        const item = Object.assign({}, orig);
        const { prob, pred } = evaluateBooster(item);
        item.risk_pred = pred;
        item.is_high_risk = pred === 1;
        item.risk_probability = Number((prob * 100).toFixed(1));
        item.status_label = pred === 1 ? "🚨 High Risk" : "✅ Stable";
        if (pred === 1) highCount++;
        results.push(item);
    }

    const total = results.length || 1;
    return {
        status: "success",
        total_evaluated: results.length,
        high_risk_count: highCount,
        stable_count: results.length - highCount,
        risk_rate_pct: Number(((highCount / total) * 100).toFixed(1)),
        records: results
    };
}

function getFeatureImportance() {
    const metricsPath = resolveFile('training_metrics.json');
    let imps = { avg_price_creep: 1.0, rec_rejection_pct: 0.0, rejection_spike_ratio: 0.0, min_days_to_expiry: 0.0, criticality_numeric: 0.0, stock_cover_days: 0.0, is_single_source: 0.0 };
    if (fs.existsSync(metricsPath)) {
        try {
            const data = JSON.parse(fs.readFileSync(metricsPath, 'utf-8'));
            if (data.feature_importances) imps = data.feature_importances;
        } catch (e) {}
    }

    const total = Object.values(imps).reduce((a, b) => a + b, 0) || 1.0;
    const featuresList = Object.entries(imps).map(([feature, importance]) => ({
        feature,
        importance: Number(importance.toFixed(4)),
        percentage: Number(((importance / total) * 100).toFixed(1))
    }));
    featuresList.sort((a, b) => b.importance - a.importance);

    return { status: "success", features: featuresList };
}

function whatIfSensitivity(baseData, featureName, steps = 10) {
    const ranges = {
        avg_price_creep: [-0.1, 1.2],
        rec_rejection_pct: [0.0, 0.5],
        rejection_spike_ratio: [0.5, 6.0],
        min_days_to_expiry: [10, 360],
        criticality_numeric: [1.0, 5.0],
        stock_cover_days: [2.0, 90.0],
        is_single_source: [0, 1]
    };

    const [rMin, rMax] = ranges[featureName] || [0.0, 1.0];
    const curve = [];

    for (let i = 0; i < steps; i++) {
        const val = steps === 1 ? rMin : rMin + (i * (rMax - rMin)) / (steps - 1);
        const simRow = Object.assign({}, baseData);
        simRow[featureName] = val;
        const { prob } = evaluateBooster(simRow);
        curve.push({
            val: Number(val.toFixed(3)),
            risk_prob: Number((prob * 100).toFixed(1))
        });
    }

    return {
        status: "success",
        feature: featureName,
        curve
    };
}

function ragQuery(query, suppliers = [], pdfKnowledge = [], topK = 4) {
    const qTokens = (query || "").toLowerCase().split(/\s+/).filter(t => t.length > 2);
    const scoredDocs = [];

    // Search Indian suppliers
    for (const s of suppliers) {
        const text = `${s.supplier_id} ${s.supplier_name} India ${s.category} ${s.region} ${s.spend_tier}`.toLowerCase();
        let matches = 0;
        for (const token of qTokens) {
            if (text.includes(token)) matches++;
        }
        if (matches > 0 || qTokens.length === 0) {
            const { pred } = evaluateBooster(s);
            scoredDocs.push({
                score: Number((matches / Math.max(qTokens.length, 1)).toFixed(2)),
                type: 'supplier',
                id: s.supplier_id,
                name: s.supplier_name,
                category: s.category,
                region: s.region,
                price_creep: Number(s.avg_price_creep || 0),
                rejection_pct: Number(s.rec_rejection_pct || 0),
                stock_cover: Number(s.stock_cover_days || 0),
                is_single_source: Number(s.is_single_source || 0),
                target: Number(s.target || 0),
                risk_pred: pred,
                doc_text: `Supplier: ${s.supplier_name} | Hub: ${s.region} | Category: ${s.category} | Price Creep: ${s.avg_price_creep} | Risk: ${pred === 1 ? 'High Risk' : 'Stable'}`
            });
        }
    }

    // Search PDF chunks
    for (const item of pdfKnowledge) {
        const text = `${item.filename} ${item.text || ''}`.toLowerCase();
        let matches = 0;
        for (const token of qTokens) {
            if (text.includes(token)) matches++;
        }
        if (matches > 0) {
            scoredDocs.push({
                score: Number((matches / Math.max(qTokens.length, 1)).toFixed(2)),
                type: 'pdf_document',
                filename: item.filename,
                chunk_id: item.chunk_id,
                title: item.title || item.filename,
                risk_pred: -1,
                doc_text: `[DOCUMENT: ${item.filename}] ${item.text}`
            });
        }
    }

    scoredDocs.sort((a, b) => b.score - a.score);
    const topResults = scoredDocs.slice(0, topK);

    return {
        status: "success",
        query: query,
        total_knowledge_docs: scoredDocs.length,
        results: topResults
    };
}

module.exports = {
    FEATURES,
    evaluateBooster,
    predictSingle,
    batchPredict,
    getFeatureImportance,
    whatIfSensitivity,
    ragQuery
};
