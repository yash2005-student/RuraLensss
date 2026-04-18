import fs from 'fs/promises';
import path from 'path';
import mongoose from 'mongoose';
import dotenv from 'dotenv';

import Scheme from '../models/Scheme.js';
import AnonymousReport from '../models/AnonymousReport.js';

dotenv.config({ path: path.resolve(process.cwd(), '.env') });

const OUTPUT_DIR = path.resolve(process.cwd(), 'pathway-data');

function safeText(value) {
  if (value === undefined || value === null) return '';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function schemeToDoc(scheme) {
  return [
    `Scheme ID: ${scheme.id}`,
    `Name: ${scheme.name}`,
    `Category: ${scheme.category || ''}`,
    `Village: ${scheme.village || ''}`,
    `District: ${scheme.district || ''}`,
    `Status: ${scheme.status || ''}`,
    `Total Budget: ${scheme.totalBudget || 0}`,
    `Budget Utilized: ${scheme.budgetUtilized || 0}`,
    `Overall Progress: ${scheme.overallProgress || 0}%`,
    `Start Date: ${scheme.startDate || ''}`,
    `End Date: ${scheme.endDate || ''}`,
    `Description: ${scheme.description || ''}`,
    `Phases: ${safeText(scheme.phases || [])}`,
    `Vendor Reports: ${safeText(scheme.vendorReports || [])}`,
    `Discrepancies: ${safeText(scheme.discrepancies || [])}`,
  ].join('\n');
}

function reportToDoc(report) {
  const c = report.anonymizedContent || {};
  return [
    `Report ID: ${report.id}`,
    `Status: ${report.status || ''}`,
    `Priority: ${report.priority || ''}`,
    `Category: ${c.problemCategory || ''}`,
    `Severity: ${c.severity || ''}`,
    `Title: ${c.title || ''}`,
    `Description: ${c.description || ''}`,
    `Intent: ${c.extractedIntent || ''}`,
    `Area: ${report.location?.area || ''}`,
    `District: ${report.location?.district || ''}`,
    `Credibility: ${report.credibilityScore || 0}`,
    `Created At: ${report.createdAt || ''}`,
  ].join('\n');
}

async function clearOutputDir() {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  const files = await fs.readdir(OUTPUT_DIR);
  for (const f of files) {
    await fs.rm(path.join(OUTPUT_DIR, f), { force: true });
  }
}

async function main() {
  const mongoUri = process.env.MONGODB_URI;
  if (!mongoUri) {
    throw new Error('MONGODB_URI is not set in backend/.env');
  }

  await mongoose.connect(mongoUri);
  await clearOutputDir();

  const schemes = await Scheme.find({}).lean();
  const reports = await AnonymousReport.find({}).lean();

  let count = 0;

  for (const s of schemes) {
    const fileName = `scheme_${s.id || count}.txt`;
    await fs.writeFile(path.join(OUTPUT_DIR, fileName), schemeToDoc(s), 'utf8');
    count += 1;
  }

  for (const r of reports) {
    const fileName = `citizen_report_${r.id || count}.txt`;
    await fs.writeFile(path.join(OUTPUT_DIR, fileName), reportToDoc(r), 'utf8');
    count += 1;
  }

  // Helpful seed file for quick smoke tests.
  await fs.writeFile(
    path.join(OUTPUT_DIR, 'system_context.txt'),
    'RuraLens pathway index generated successfully. Use questions about schemes, budgets, discrepancies, and citizen reports.',
    'utf8'
  );

  await mongoose.disconnect();
  console.log(`✅ Exported ${count} docs to ${OUTPUT_DIR}`);
}

main().catch((err) => {
  console.error('❌ export-to-pathway failed:', err.message);
  process.exit(1);
});
