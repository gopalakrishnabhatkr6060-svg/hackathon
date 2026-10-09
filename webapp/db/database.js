const fs = require('fs');
const path = require('path');

const DB_FILE = path.join(__dirname, 'suppliers.json');
const CSV_FILE = path.join(__dirname, '..', '..', 'suppliers_dataset.csv');

function parseCSVLine(line) {
    const values = [];
    let insideQuotes = false;
    let currentVal = '';
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') {
            insideQuotes = !insideQuotes;
        } else if (char === ',' && !insideQuotes) {
            values.push(currentVal.trim());
            currentVal = '';
        } else {
            currentVal += char;
        }
    }
    values.push(currentVal.trim());
    return values.map(v => v.replace(/^"|"$/g, '').trim());
}

// Initialize database
function initDatabase() {
    if (!fs.existsSync(DB_FILE)) {
        console.log('[DB] Initializing database from suppliers_dataset.csv...');
        if (fs.existsSync(CSV_FILE)) {
            const rawCSV = fs.readFileSync(CSV_FILE, 'utf-8');
            const lines = rawCSV.trim().split('\n');
            const headers = parseCSVLine(lines[0]);
            
            const records = [];
            for (let i = 1; i < lines.length; i++) {
                if (!lines[i].trim()) continue;
                const vals = parseCSVLine(lines[i]);
                const obj = {};
                headers.forEach((h, idx) => {
                    let val = vals[idx] !== undefined ? vals[idx] : '';
                    if (!isNaN(val) && val !== '') {
                        val = Number(val);
                    }
                    obj[h] = val;
                });
                records.push(obj);
            }
            fs.writeFileSync(DB_FILE, JSON.stringify(records, null, 2), 'utf-8');
            console.log(`[DB] Seeded ${records.length} suppliers into database.`);
        } else {
            fs.writeFileSync(DB_FILE, JSON.stringify([], null, 2), 'utf-8');
        }
    }
}

function getAllSuppliers() {
    initDatabase();
    try {
        const data = fs.readFileSync(DB_FILE, 'utf-8');
        return JSON.parse(data);
    } catch (e) {
        return [];
    }
}

function getSupplierById(id) {
    const list = getAllSuppliers();
    return list.find(s => s.supplier_id === id) || null;
}

function createSupplier(newSupplier) {
    const list = getAllSuppliers();
    const id = newSupplier.supplier_id || `SUP-${1000 + list.length + 1}`;
    const record = { ...newSupplier, supplier_id: id, created_at: new Date().toISOString() };
    list.unshift(record);
    fs.writeFileSync(DB_FILE, JSON.stringify(list, null, 2), 'utf-8');
    return record;
}

function getDatabaseStats() {
    const list = getAllSuppliers();
    const categories = new Set(list.map(s => s.category));
    const regions = new Set(list.map(s => s.region));
    const singleSourceCount = list.filter(s => s.is_single_source === 1).length;
    
    return {
        totalSuppliers: list.length,
        totalCategories: categories.size,
        totalRegions: regions.size,
        singleSourceCount: singleSourceCount
    };
}

module.exports = {
    initDatabase,
    getAllSuppliers,
    getSupplierById,
    createSupplier,
    getDatabaseStats,
    parseCSVLine
};
