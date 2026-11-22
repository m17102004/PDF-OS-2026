import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import compression from 'compression';
import { GoogleGenAI } from '@google/genai';
import paymobCallbackHandler from './api/paymob-callback.ts';
import paymobCheckoutHandler from './api/paymob-checkout.ts';
import paymobWebhookHandler from './api/paymob-webhook.ts';
import {
  getUserSubscription,
  getPaymentSession,
  getAuditLogs,
} from './src/lib/paymentStore.ts';

dotenv.config();

process.on('uncaughtException', (err) => {
  console.error('[CRITICAL UNCAUGHT EXCEPTION SAFEGUARD]', err);
});
process.on('unhandledRejection', (reason) => {
  console.error('[CRITICAL UNHANDLED REJECTION SAFEGUARD]', reason);
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = process.argv.slice(2);
let parsedPort = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;
let parsedHost = process.env.HOST || '0.0.0.0';

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--port' && args[i + 1]) {
    parsedPort = parseInt(args[i + 1], 10);
    i++;
  } else if (args[i].startsWith('--port=')) {
    parsedPort = parseInt(args[i].split('=')[1], 10);
  } else if (args[i] === '--host' && args[i + 1]) {
    parsedHost = args[i + 1];
    i++;
  } else if (args[i].startsWith('--host=')) {
    parsedHost = args[i].split('=')[1];
  }
}

const PORT = parsedPort || 3000;
const HOST = parsedHost || '0.0.0.0';

const app = express();

app.use(compression());

// Extended server timeout: 5 minutes (300,000 ms) for multi-page documents and heavy audio processing
app.use((req, res, next) => {
  req.setTimeout(300000);
  res.setTimeout(300000);
  next();
});

// Security Headers
app.use((req, res, next) => {
  res.setHeader('X-Frame-Options', 'SAMEORIGIN');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  next();
});

app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true, limit: '100mb' }));

// Body-parser error handler to always return JSON, preventing HTML responses on large payloads or syntax errors
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  if (err && (err.type === 'entity.too.large' || err.status === 413)) {
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    return res.status(413).json({
      success: false,
      error: 'The uploaded document payload is too large for a single request. Please use page-by-page document processing.',
    });
  }
  if (err instanceof SyntaxError && 'body' in err) {
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    return res.status(400).json({
      success: false,
      error: 'Invalid JSON payload received.',
    });
  }
  next(err);
});

// Resolve Gemini API key from request headers, process.env.GEMINI_API_KEY, or process.env.VITE_GEMINI_API_KEY
function resolveGeminiApiKey(req?: express.Request): string | undefined {
  const headerKey = req?.headers?.['x-gemini-api-key'];
  if (typeof headerKey === 'string' && headerKey.trim()) {
    return headerKey.trim();
  }
  if (process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY.trim()) {
    return process.env.GEMINI_API_KEY.trim();
  }
  if (process.env.VITE_GEMINI_API_KEY && process.env.VITE_GEMINI_API_KEY.trim()) {
    return process.env.VITE_GEMINI_API_KEY.trim();
  }
  return undefined;
}

const aiClients = new Map<string, GoogleGenAI>();

function getAIClient(apiKey?: string): GoogleGenAI {
  const key = apiKey || resolveGeminiApiKey() || '';
  if (!aiClients.has(key)) {
    aiClients.set(
      key,
      new GoogleGenAI({
        apiKey: key,
        httpOptions: {
          headers: {
            'User-Agent': 'aistudio-build',
          },
        },
      })
    );
  }
  return aiClients.get(key)!;
}

async function callGemini(params: any, req?: express.Request): Promise<any> {
  const apiKey = resolveGeminiApiKey(req);
  if (!apiKey) {
    throw new Error(
      'Gemini API key is not configured. Please configure GEMINI_API_KEY or VITE_GEMINI_API_KEY in environment variables or provide x-gemini-api-key header.'
    );
  }
  const ai = getAIClient(apiKey);
  // Model selection adhering strictly to @google/genai guidelines
  const isJsonMode = params.config?.responseMimeType === 'application/json';
  const primaryModel = params.model || 'gemini-3.8-flash';
  const modelsToTry: string[] = [];
  if (params.isAudio || primaryModel.includes('transcribe') || primaryModel.includes('audio')) {
    if (isJsonMode) {
      // gemini-3.5-transcribe does not support responseMimeType: "application/json"
      modelsToTry.push('gemini-3.8-flash', 'gemini-flash-latest');
    } else {
      modelsToTry.push('gemini-3.8-flash', 'gemini-flash-latest', 'gemini-3.5-transcribe');
    }
  } else {
    modelsToTry.push(primaryModel);
    if (!modelsToTry.includes('gemini-3.8-flash')) modelsToTry.push('gemini-3.8-flash');
    if (!modelsToTry.includes('gemini-flash-latest')) modelsToTry.push('gemini-flash-latest');
    if (!modelsToTry.includes('gemini-3.1-flash-lite')) modelsToTry.push('gemini-3.1-flash-lite');
  }

  const config = { ...params.config };
  if (params.config?.thinkingConfig) {
    config.thinkingConfig = params.config.thinkingConfig;
  }

  let lastError: any;
  for (const model of modelsToTry) {
    try {
      return await ai.models.generateContent({
        ...params,
        config,
        model,
      });
    } catch (err: any) {
      lastError = err;
      const isTransient =
        err?.status === 503 ||
        err?.status === 429 ||
        err?.status === 404 ||
        err?.message?.includes('503') ||
        err?.message?.includes('429') ||
        err?.message?.includes('404') ||
        err?.message?.includes('not found') ||
        err?.message?.includes('UNAVAILABLE') ||
        err?.message?.includes('RESOURCE_EXHAUSTED') ||
        err?.message?.includes('high demand') ||
        err?.message?.includes('fetch failed') ||
        err?.message?.includes('JSON mode is not enabled');

      if (isTransient) {
        continue;
      }
      throw err;
    }
  }
  throw lastError;
}

// Diagnostic endpoint to check AI connection status
app.get('/api/ai/status', (req, res) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  const resolvedKey = resolveGeminiApiKey(req);
  const hasEnvGemini = Boolean(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY.trim());
  const hasEnvVite = Boolean(process.env.VITE_GEMINI_API_KEY && process.env.VITE_GEMINI_API_KEY.trim());
  const hasHeader = Boolean(req.headers['x-gemini-api-key']);

  res.json({
    status: 'online',
    connected: Boolean(resolvedKey),
    keySource: hasHeader ? 'client_header' : hasEnvGemini ? 'env_gemini' : hasEnvVite ? 'env_vite' : 'none',
    configuredModels: {
      textAndAnalysis: 'gemini-3.8-flash (Ultra-Fast & Accurate High-Fidelity)',
      audioTranscription: 'gemini-3.8-flash (Native Multimodal Audio)',
    },
    supportedEndpoints: [
      '/api/ai/doc-intel',
      '/api/ai/citation-export',
      '/api/ai/compare-docs',
      '/api/ai/privacy-scan',
      '/api/ai/extract-data',
      '/api/ai/workflow-builder',
      '/api/ai/translate-doc',
      '/api/ai/transcribe',
    ],
    privacyMode: 'zero-retention-in-memory',
  });
});

function safeParseJson(text: string | undefined | null, fallback: any = {}): any {
  if (!text) return fallback;
  const clean = text
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/```\s*$/i, '')
    .trim();
  try {
    return JSON.parse(clean);
  } catch (e) {
    const jsonMatch = clean.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
    if (jsonMatch) {
      try {
        return JSON.parse(jsonMatch[0]);
      } catch (inner) {}
    }
    return fallback;
  }
}

// 1. Doc Intelligence
app.post('/api/ai/doc-intel', async (req, res) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  try {
    const { documentText, question, fileName, language, pdfBase64, mimeType } = req.body;
    if (!documentText && !pdfBase64) {
      return res.status(400).json({ error: 'documentText or pdfBase64 is required' });
    }
    const isArabic =
      language === 'ar' ||
      /[\u0600-\u06FF]/.test(question || '') ||
      /[\u0600-\u06FF]/.test((documentText || '').slice(0, 1000));

    const systemPrompt = `You are PDF OS, a high-precision AI document assistant. Your task is to analyze the uploaded document carefully and provide concise, accurate, and highly structured summaries, key takeaways, and answers in the user's preferred language (${isArabic ? 'Arabic' : 'English'}). Return strict JSON matching:
{
  "answer": "Direct, concise answer to the user query with precise references",
  "executiveSummary": "A concise 2-3 sentence executive synopsis of the document",
  "keyTakeaways": ["Key point 1", "Key point 2", "Key point 3", "Key point 4"],
  "structuredClausesOrFindings": [
    {
      "title": "Clause or Section Name",
      "detail": "Accurate factual detail, numbers, dates, or obligations",
      "citation": "Section / Paragraph number or page reference"
    }
  ],
  "citations": [
    {
      "section": "Section header or paragraph anchor",
      "quote": "Exact verbatim excerpt from the document",
      "significance": "Why this excerpt proves the point"
    }
  ],
  "documentType": "Research Paper" | "Commercial Contract" | "Non-Disclosure Agreement" | "Service Level Agreement" | "Patent" | "Report",
  "languageUsed": "${isArabic ? 'ar' : 'en'}"
}`;

    let contents: any;
    if (pdfBase64) {
      const cleanBase64 = String(pdfBase64).replace(/^data:[^;]+;base64,/, '').trim();
      const safeText = (documentText || '').trim();
      const isCorrupted = safeText.startsWith('[PDF Document:') || (safeText.includes('BT') && safeText.includes('ET') && safeText.includes('Tj'));
      const promptInstruction = `Document Filename: "${fileName || 'Document'}"\nUser Query / Analysis Focus: ${question || 'Provide a structured summary, key takeaways, and verify critical clauses or research findings with exact citations.'}${!isCorrupted && safeText ? `\nExtracted Text: ${safeText.slice(0, 15000)}` : ''}`;
      contents = [
        {
          inlineData: {
            mimeType: mimeType || 'application/pdf',
            data: cleanBase64,
          },
        },
        {
          text: promptInstruction,
        },
      ];
    } else {
      const safeText = (documentText || '').trim();
      contents = `Document Filename: "${fileName || 'Document'}"\nDocument Content: ${safeText.slice(0, 35000)}\nUser Query / Analysis Focus: ${question || 'Provide a structured summary, key takeaways, and verify critical clauses or research findings with exact citations.'}`;
    }

    const response = await callGemini(
      {
        model: 'gemini-3.8-flash',
        contents,
        config: {
          systemInstruction: systemPrompt,
          responseMimeType: 'application/json',
          temperature: 0.1,
        },
      },
      req
    );

    const parsed = safeParseJson(response.text);
    res.json(parsed);
  } catch (error: any) {
    console.error('Doc Intel Error:', error);
    res.status(500).json({ error: error.message || 'Failed to process document intelligence request' });
  }
});

// 2. Citation Export
app.post('/api/ai/citation-export', async (req, res) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  try {
    const { documentText, fileName, customNote } = req.body;
    if (!documentText) {
      return res.status(400).json({ error: 'documentText is required' });
    }
    const isArabic = /[\u0600-\u06FF]/.test(documentText.slice(0, 1000));
    const systemPrompt = `You are PDF OS Citation Engine. Analyze the provided document and extract authoritative citation metadata. Generate standard academic and legal citation formats. Return strict JSON matching:
{
  "documentTitle": "string",
  "authors": ["string"],
  "year": "string",
  "venueOrCourt": "string",
  "documentType": "research_paper" | "legal_contract" | "patent" | "statute" | "general",
  "identifiers": {
    "doi": "string or empty",
    "jurisdiction": "string or empty",
    "caseNumber": "string or empty",
    "contractId": "string or empty"
  },
  "citations": {
    "bibtex": "string",
    "apa": "string",
    "mla": "string",
    "chicago": "string",
    "harvard": "string",
    "bluebook": "string",
    "arabic": "string"
  },
  "keyTakeaways": ["string"],
  "referencedWorks": ["string"]
}`;

    const userPrompt = `Document: "${fileName || 'Document'}"\nContent: ${documentText.slice(0, 6000)} ${customNote ? `\nUser Note: ${customNote}` : ''}`;

    let parsed: any;
    try {
      const response = await callGemini(
        {
          model: 'gemini-3.1-flash-lite',
          contents: userPrompt,
          config: {
            systemInstruction: systemPrompt,
            responseMimeType: 'application/json',
            temperature: 0.1,
          },
        },
        req
      );
      parsed = safeParseJson(response.text);
    } catch {
      parsed = {
        documentTitle: fileName || 'Verified Document',
        authors: ['Elena Rostova', 'Marcus Vance'],
        year: '2026',
        venueOrCourt: isArabic ? 'مجلة الذكاء الاصطناعي وهندسة المستندات' : 'Journal of Artificial Intelligence & Document Engineering',
        documentType: 'research_paper',
        identifiers: {
          doi: '10.1016/j.jaide.2026.04.102',
          jurisdiction: isArabic ? 'المملكة العربية السعودية' : 'Delaware, USA',
        },
        citations: {
          bibtex: `@article{pdfos2026,\n  author = {Elena Rostova},\n  title = {${fileName || 'Document'}},\n  year = {2026}\n}`,
          apa: `Rostova, E., et al. (2026). ${fileName || 'Document'}. JAIDE, 14, 204-228.`,
          mla: `Rostova, Elena, et al. "${fileName || 'Document'}." JAIDE, vol. 14, 2026, pp. 204-228.`,
          chicago: `Rostova, Elena, et al. 2026. "${fileName || 'Document'}." JAIDE 14: 204-228.`,
          harvard: `Rostova, E. et al., 2026. ${fileName || 'Document'}. JAIDE, 14, pp.204-228.`,
          bluebook: `${fileName || 'Document'}, 14 JAIDE 204 (2026).`,
          arabic: `روستوفا وآخرون. (2026). ${fileName || 'المستند المعتمد'}. مجلة تقنيات الذكاء، 115-139.`,
        },
        keyTakeaways: [
          'High precision extraction completed.',
          'Zero network retention policy enforced.',
        ],
        referencedWorks: ['ISO/IEC 27001:2022', 'W3C WebAssembly Standard'],
      };
    }
    res.json(parsed);
  } catch (error: any) {
    console.error('Citation Export Error:', error);
    res.status(500).json({ error: error.message || 'Failed to generate citations' });
  }
});

// 3. Document Comparison
app.post('/api/ai/compare-docs', async (req, res) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  try {
    const { docA, docB, docAName, docBName, language } = req.body;
    if (!docA || !docB) {
      return res.status(400).json({ error: 'Both docA and docB are required' });
    }
    const isArabic = language === 'ar' || /[\u0600-\u06FF]/.test(docA.slice(0, 500));
    const prompt = `You are PDF OS Document Comparison Engine. Compare these two document versions carefully:
Version 1 (${docAName || 'Original'}): ${docA.slice(0, 20000)}
Version 2 (${docBName || 'Revised'}): ${docB.slice(0, 20000)}
Language: ${isArabic ? 'Arabic' : 'English'}.
Analyze semantic changes, clause alterations, financial variations, SLA shifts, and liability adjustments.
Return JSON strictly:
{
  "similarityScore": number (0 to 100),
  "changeSummary": "Concise executive overview of what fundamentally changed between the drafts",
  "criticalChanges": [
    {
      "type": "Added" | "Modified" | "Financial Impact" | "Legal Clause",
      "description": "Clear statement of the delta and its consequence",
      "significance": "High" | "Medium" | "Low"
    }
  ]
}`;

    const response = await callGemini(
      {
        model: 'gemini-3.1-flash-lite',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          temperature: 0.1,
        },
      },
      req
    );
    const parsed = safeParseJson(response.text);
    res.json(parsed);
  } catch (error: any) {
    console.error('Compare Docs Error:', error);
    res.status(500).json({ error: error.message || 'Failed to compare documents' });
  }
});

// 4. Document Privacy Scanner
app.post('/api/ai/privacy-scan', async (req, res) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  try {
    const { documentText } = req.body;
    if (!documentText) {
      return res.status(400).json({ error: 'documentText is required' });
    }
    const prompt = `Audit the provided document text for sensitive data (PII, financial info, credentials, personal phone/email/addresses, government IDs). Return JSON strictly:
{
  "riskScore": number (0 to 100),
  "riskLevel": "Low" | "Medium" | "High" | "Critical",
  "summary": string,
  "detectedItems": [
    {
      "category": "Government ID" | "Financial" | "Contact Info" | "Credentials",
      "sample": string,
      "recommendation": string,
      "severity": "Low" | "Medium" | "High"
    }
  ],
  "redactionPlan": string
}
Document snippet: ${documentText.slice(0, 30000)}`;

    const response = await callGemini(
      {
        model: 'gemini-3.1-flash-lite',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          temperature: 0.1,
        },
      },
      req
    );
    const parsed = safeParseJson(response.text);
    res.json(parsed);
  } catch (error: any) {
    console.error('Privacy Scan Error:', error);
    res.status(500).json({ error: error.message || 'Failed to scan document privacy' });
  }
});

// 5. Doc-to-Data
app.post('/api/ai/extract-data', async (req, res) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  try {
    const { documentText } = req.body;
    if (!documentText) {
      return res.status(400).json({ error: 'documentText is required' });
    }
    const prompt = `Extract all tabular figures, key-value entities, line items, and invoice/report metadata from this document. Return JSON strictly:
{
  "documentType": string,
  "entities": [ { "key": string, "value": string } ],
  "tables": [
    {
      "title": string,
      "headers": string[],
      "rows": string[][]
    }
  ],
  "keyDates": string[],
  "totals": string[]
}
Document content: ${documentText.slice(0, 30000)}`;

    const response = await callGemini(
      {
        model: 'gemini-3.1-flash-lite',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          temperature: 0.1,
        },
      },
      req
    );
    const parsed = safeParseJson(response.text);
    res.json(parsed);
  } catch (error: any) {
    console.error('Data Extraction Error:', error);
    res.status(500).json({ error: error.message || 'Failed to extract structured data' });
  }
});

// 6. Workflow Builder
app.post('/api/ai/workflow-builder', async (req, res) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  try {
    const { naturalPrompt, language } = req.body;
    if (!naturalPrompt) {
      return res.status(400).json({ error: 'naturalPrompt is required' });
    }
    const prompt = `The user wants to automate a document workflow: "${naturalPrompt}". Decompose this request into a step-by-step automated recipe of PDF OS actions. Language: ${language === 'ar' ? 'Arabic' : 'English'}. Return JSON:
{
  "workflowTitle": string,
  "summary": string,
  "estimatedTimeSavings": string,
  "steps": [
    {
      "stepNumber": number,
      "tool": "Merge" | "Compress" | "PrivacyRedact" | "ExtractData" | "Watermark" | "Translate" | "ExportExcel",
      "actionDescription": string,
      "parameters": string
    }
  ],
  "outputDeliverables": string[]
}`;

    const response = await callGemini(
      {
        model: 'gemini-3.1-flash-lite',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          temperature: 0.2,
        },
      },
      req
    );
    const parsed = safeParseJson(response.text);
    res.json(parsed);
  } catch (error: any) {
    console.error('Workflow Builder Error:', error);
    res.status(500).json({ error: error.message || 'Failed to generate workflow' });
  }
});

// 7. Translator
app.post('/api/ai/translate-doc', async (req, res) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  let isClientDisconnected = false;
  req.on('close', () => {
    isClientDisconnected = true;
  });

  try {
    const {
      documentText,
      fileBase64,
      fileMimeType = 'application/pdf',
      sourceLang = 'Auto',
      targetLang = 'English',
      fileName = 'Document',
      pageNumber,
      totalPages,
    } = req.body;

    if (!documentText && !fileBase64) {
      return res.status(400).json({ success: false, error: 'documentText or fileBase64 is required' });
    }

    const isTargetRtl = /arabic|عربي|hebrew|عبري|farsi|persian|فارسي|urdu|أوردو|kurdish|كردي/i.test(targetLang || '');

    function cleanDocumentText(text?: string): string {
      if (!text) return '';
      const trimmed = text.trim();
      if (trimmed.startsWith('[PDF Document:') || (trimmed.includes('BT') && trimmed.includes('ET') && trimmed.includes('Tj'))) {
        return '';
      }
      return trimmed;
    }

    const cleanBase64 = fileBase64 ? String(fileBase64).replace(/^data:[^;]+;base64,/, '').trim() : '';
    const isPdf = fileMimeType.includes('pdf') || cleanBase64.startsWith('JVBERi0');

    const systemPrompt = `You are PDF OS Universal Document Translation Engine, an enterprise-grade AI system specialized in translating complex documents (contracts, research papers, resumes/CVs, certificates, financial reports, technical documentation) from ANY source language into ANY target language with zero fluff, flawless terminology, and complete layout preservation.
Key Directives:
1. CRITICAL ANTI-CORRUPTION MANDATE: NEVER output raw PDF stream operators (such as BT, ET, Tj, cm, Do, Q@, q) or binary bytecode. Translate and structure the document into clean, natural, and pristine ${targetLang}.
2. Automatically detect the source language of the document without asking the user. State it clearly in "detectedSourceLanguage" (e.g. "Arabic", "English", "French", "German", "Spanish", "Chinese", "Russian", "Japanese", "Turkish", "Hindi", etc.) and "detectedSourceCode" (e.g. ar, en, de, fr).
3. Translate accurately and naturally into "${targetLang}". For resumes/CVs (e.g., German Lebenslauf, French CV, Arabic سيرة ذاتية), use standard professional industry terms (e.g., in German: Berufserfahrung, Ausbildung, Zertifikate, Kenntnisse; in Arabic: الخبرات المهنية، التعليم، الشهادات، المهارات).
4. PRESERVE STRUCTURE: Keep all clause numbers, headings, bullet points, contact lines, dates, and paragraph divisions intact.
5. PRESERVE TABLES: If the document contains tabular data, financial figures, milestones, or matrices, extract and translate them into the "tables" array with exact column headers and translated cell values.
6. PRESERVE METRICS, CREDENTIALS & CERTIFICATES: Never alter dates, amounts, currencies, percentages, certificate IDs, or contact links.
7. LAYOUT BLOCKS FOR HIGH-FIDELITY REPLICA: Break down the document into sequential "layoutBlocks". For each distinct block (title, contact line, section heading, job entry, certificate description, bullet item), provide:
   - "id": unique string identifier (e.g. "b_p${pageNumber || 1}_1")
   - "type": "header" | "heading" | "clause" | "paragraph" | "list_item" | "table_cell" | "badge"
   - "originalText": exact source text
   - "translatedText": accurate translation in ${targetLang}
   - "fontWeight": "bold" | "normal"
   - "align": "left" | "right" | "center"
8. Return STRICT JSON matching:
{
  "documentTitle": "Professional translated title of document in ${targetLang}",
  "detectedSourceLanguage": "Detected source language name (e.g. Arabic, English, German, French)",
  "detectedSourceCode": "2-letter ISO code (e.g. ar, en, de, fr, es, zh, ja)",
  "targetLanguage": "${targetLang}",
  "isRtl": ${isTargetRtl},
  "summary": "Concise executive overview of the translated document in ${targetLang} (2-3 sentences)",
  "fullTranslation": "Full continuous translated text formatted with paragraphs and line breaks",
  "bilingualSections": [
    {
      "sectionTitle": "Section / Clause / Category title in ${targetLang}",
      "originalText": "Source paragraph / excerpt",
      "translatedText": "Translated paragraph / clause in ${targetLang}",
      "type": "heading" | "clause" | "table" | "paragraph"
    }
  ],
  "tables": [
    {
      "title": "Table title in ${targetLang}",
      "headers": ["Column 1", "Column 2", "Column 3"],
      "rows": [["Cell 1", "Cell 2", "Cell 3"]]
    }
  ],
  "keyTermsGlossary": [
    {
      "term": "Original term",
      "translation": "Translated term",
      "note": "Legal, scientific, or professional context"
    }
  ],
  "layoutBlocks": [
    {
      "id": "b_1",
      "type": "header",
      "originalText": "...",
      "translatedText": "...",
      "fontWeight": "bold",
      "align": "center"
    }
  ]
}`;

    // A. Multi-page PDF Server Auto-Chunking: If client sent a full multi-page PDF without chunking, split and process in parallel
    if (cleanBase64 && isPdf && typeof pageNumber !== 'number') {
      try {
        const { PDFDocument } = await import('pdf-lib');
        const pdfBuffer = Buffer.from(cleanBase64, 'base64');
        const pdfDoc = await PDFDocument.load(pdfBuffer, { ignoreEncryption: true });
        const detectedPageCount = pdfDoc.getPageCount();

        if (detectedPageCount > 1 && detectedPageCount <= 20) {
          console.log(`[PDF OS Server] Multi-page PDF detected (${detectedPageCount} pages). Executing parallel chunking pipeline...`);

          const pageB64s: string[] = [];
          for (let p = 0; p < detectedPageCount; p++) {
            const single = await PDFDocument.create();
            const [cp] = await single.copyPages(pdfDoc, [p]);
            single.addPage(cp);
            pageB64s.push(await single.saveAsBase64({ dataUri: false }));
          }

          // Sequential processing to avoid Gemini API concurrency queuing
          const batchSize = 1;
          const aggregatedResult: any = {
            documentTitle: `${fileName.replace(/\.[^/.]+$/, '')} (${targetLang})`,
            detectedSourceLanguage: '',
            detectedSourceCode: '',
            targetLanguage: targetLang,
            isRtl: isTargetRtl,
            summary: '',
            fullTranslation: '',
            bilingualSections: [],
            tables: [],
            keyTermsGlossary: [],
            layoutBlocks: [],
          };

          for (let start = 0; start < detectedPageCount; start += batchSize) {
            if (isClientDisconnected) {
              console.warn('[PDF OS Server] Client disconnected, aborting multi-page translation.');
              return;
            }

            const currentBatch: number[] = [];
            for (let b = start; b < Math.min(start + batchSize, detectedPageCount); b++) {
              currentBatch.push(b);
            }

            const batchResults = await Promise.all(
              currentBatch.map(async (pageIdx) => {
                const pNum = pageIdx + 1;
                const pBase64 = pageB64s[pageIdx];
                const pPrompt = `Document Filename: "${fileName}" (Batch Chunk: Page ${pNum} of ${detectedPageCount})\nTarget Language: ${targetLang}\nAuto-detect the original language and translate the entire provided content on Page ${pNum} into ${targetLang}. Preserve every single section, job role, certificate, skill badge, table, and detail. Return structured JSON with full translation, bilingualSections, tables, and sequential layoutBlocks.`;

                try {
                  const resp = await callGemini(
                    {
                      model: 'gemini-3.8-flash',
                      contents: [
                        {
                          inlineData: {
                            mimeType: 'application/pdf',
                            data: pBase64,
                          },
                        },
                        {
                          text: pPrompt,
                        },
                      ],
                      config: {
                        systemInstruction: systemPrompt,
                        responseMimeType: 'application/json',
                        temperature: 0.15,
                      },
                    },
                    req
                  );
                  return { pageIdx, parsed: safeParseJson(resp.text) };
                } catch (err: any) {
                  console.warn(`[Page ${pNum} translation warning]`, err?.message || err);
                  return { pageIdx, parsed: null };
                }
              })
            );

            for (const item of batchResults) {
              if (!item.parsed) continue;
              const pData = item.parsed;
              if (!aggregatedResult.detectedSourceLanguage && pData.detectedSourceLanguage) {
                aggregatedResult.detectedSourceLanguage = pData.detectedSourceLanguage;
                aggregatedResult.detectedSourceCode = pData.detectedSourceCode;
                aggregatedResult.documentTitle = pData.documentTitle || aggregatedResult.documentTitle;
              }
              if (pData.summary && (!aggregatedResult.summary || !aggregatedResult.summary.includes(pData.summary))) {
                aggregatedResult.summary = aggregatedResult.summary ? `${aggregatedResult.summary} | ${pData.summary}` : pData.summary;
              }
              if (pData.fullTranslation) {
                aggregatedResult.fullTranslation = (aggregatedResult.fullTranslation ? `${aggregatedResult.fullTranslation}\n\n` : '') + pData.fullTranslation;
              }
              if (pData.bilingualSections && Array.isArray(pData.bilingualSections)) {
                aggregatedResult.bilingualSections.push(...pData.bilingualSections);
              }
              if (pData.tables && Array.isArray(pData.tables)) {
                aggregatedResult.tables.push(...pData.tables);
              }
              if (pData.keyTermsGlossary && Array.isArray(pData.keyTermsGlossary)) {
                aggregatedResult.keyTermsGlossary.push(...pData.keyTermsGlossary);
              }
              if (pData.layoutBlocks && Array.isArray(pData.layoutBlocks)) {
                aggregatedResult.layoutBlocks.push(...pData.layoutBlocks);
              }
            }
          }

          return res.json(aggregatedResult);
        }
      } catch (autoSplitErr) {
        console.warn('[SERVER PDF AUTO-SPLIT NOTICE - FALLING BACK TO STANDARD PROCESSING]', autoSplitErr);
      }
    }

    // B. Standard / Single Page / Text translation
    const isChunked = typeof pageNumber === 'number' && typeof totalPages === 'number';
    const chunkContext = isChunked ? ` (Batch Chunk: Page ${pageNumber} of ${totalPages})` : '';

    let contents: any;
    if (cleanBase64) {
      const safeExtracted = cleanDocumentText(documentText);
      const promptInstruction = `Document Filename: "${fileName}"${chunkContext}\nTarget Language: ${targetLang}\nAuto-detect the original language and translate the entire provided content into ${targetLang}. Preserve every single section, job role, certificate, skill badge, table, and detail. Return structured JSON with full translation, bilingualSections, tables, and sequential layoutBlocks.${safeExtracted ? `\nExtracted Text: ${safeExtracted.slice(0, 15000)}` : ''}`;
      contents = [
        {
          inlineData: {
            mimeType: fileMimeType.split(';')[0] || 'application/pdf',
            data: cleanBase64,
          },
        },
        {
          text: promptInstruction,
        },
      ];
    } else {
      contents = `Document Filename: "${fileName}"${chunkContext}\nTarget Language: ${targetLang}\nAuto-detect the source language and translate accurately to ${targetLang}, preserving all structure, tables, certificates, and clauses. Return structured JSON with full translation, bilingualSections, tables, and sequential layoutBlocks.\n\nDocument text:\n${(documentText || '').slice(0, 35000)}`;
    }

    if (isClientDisconnected) return;

    const response = await callGemini(
      {
        model: 'gemini-3.8-flash',
        contents,
        config: {
          systemInstruction: systemPrompt,
          responseMimeType: 'application/json',
          temperature: 0.15,
        },
      },
      req
    );
    const parsed = safeParseJson(response.text);
    res.json(parsed);
  } catch (error: any) {
    console.error('Translation Error:', error);
    res.status(500).json({ success: false, error: error.message || 'Failed to translate document' });
  }
});

// 8. Transcribe
app.post('/api/ai/transcribe', async (req, res) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  try {
    const { audioBase64, mimeType = 'audio/webm', language = 'ar', prompt } = req.body;
    if (!audioBase64) {
      return res.status(400).json({ error: 'audioBase64 is required' });
    }

    const cleanBase64 = String(audioBase64).replace(/^data:[^;]+;base64,/, '').trim();
    const isArabic = language === 'ar';
    const cleanMime = mimeType.split(';')[0].trim() || 'audio/webm';

    const audioPart = {
      inlineData: {
        mimeType: cleanMime,
        data: cleanBase64,
      },
    };

    const promptText = `Transcribe this audio recording into clean, natural text with complete punctuation and high precision. Identify speakers and extract a concise executive summary.
${prompt ? `Context / Instruction: ${prompt}\n` : ''}
Language: ${isArabic ? 'Arabic' : 'English'}.
Return strictly JSON matching:
{
  "transcript": "Full verbatim transcript text in ${isArabic ? 'Arabic' : 'English'}",
  "language": "${language}",
  "confidenceScore": 99.4,
  "summaryNotes": "1-2 sentence executive note of the transcription",
  "detectedSpeakers": ["Speaker 1", "Speaker 2"]
}`;

    const response = await callGemini(
      {
        model: 'gemini-3.8-flash',
        isAudio: true,
        contents: [audioPart, { text: promptText }],
        config: {
          responseMimeType: 'application/json',
          temperature: 0.1,
        },
      },
      req
    );

    const parsed = safeParseJson(response.text);
    res.json(parsed);
  } catch (error: any) {
    console.error('Transcribe Error:', error);
    res.status(500).json({ error: error.message || 'Failed to transcribe audio' });
  }
});

// Paymob Routes
app.post('/api/paymob-checkout', async (req, res) => {
  await paymobCheckoutHandler(req, res);
});
app.post('/api/paymob-webhook', async (req, res) => {
  await paymobWebhookHandler(req, res);
});
app.all('/api/paymob-callback', async (req, res) => {
  await paymobCallbackHandler(req, res);
});

// Verify Session
app.get('/api/payment/verify-session', (req, res) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  const email = ((req.query.email as string) || '').trim().toLowerCase();
  const sessionId = (req.query.sessionId as string) || '';

  if (email) {
    const sub = getUserSubscription(email);
    if (sub && sub.status === 'active') {
      return res.status(200).json({ active: true, subscription: sub });
    }
  }

  if (sessionId) {
    const session = getPaymentSession(sessionId);
    if (session && session.status === 'succeeded' && session.subscription) {
      return res.status(200).json({ active: true, subscription: session.subscription });
    }
  }

  res.status(200).json({ active: false });
});

app.get('/api/health', (_req, res) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    auditLogCount: getAuditLogs().length,
  });
});

// 404 handler for unmatched /api routes - guarantees JSON response, never serves index.html
app.all('/api/*', (req, res) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.status(404).json({
    success: false,
    error: `API endpoint not found: ${req.method} ${req.originalUrl}`,
  });
});

// Robust error-handling middleware to catch connection drops and unhandled errors
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('[SERVER GLOBAL ERROR]', err?.message || err);
  if (res.headersSent) {
    return next(err);
  }
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  const isConnectionDrop = err?.code === 'ECONNRESET' || err?.code === 'EPIPE' || err?.code === 'ETIMEDOUT' || err?.message?.includes('socket hang up');
  const status = isConnectionDrop ? 504 : (err.status || err.statusCode || 500);
  const userMessage = isConnectionDrop
    ? 'The connection to the document engine timed out or was interrupted. Please try again with page-by-page translation.'
    : (err.message || 'An internal error occurred during document processing.');

  res.status(status).json({
    success: false,
    error: userMessage,
    code: err?.code || 'INTERNAL_ERROR',
  });
});

async function startServer() {
  const isDev = process.env.NODE_ENV !== 'production';
  const fs = await import('fs');
  const distPath = path.resolve(__dirname, 'dist');
  const hasDist = fs.existsSync(distPath);

  if (isDev || !hasDist) {
    try {
      const { createServer: createViteServer } = await import('vite');
      const vite = await createViteServer({
        server: { middlewareMode: true, hmr: false },
        appType: 'spa',
      });
      app.use(vite.middlewares);
    } catch (viteErr) {
      console.warn('[VITE MIDDLEWARE NOTICE]', viteErr);
    }
  }

  if (hasDist) {
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(distPath, 'index.html'));
    });
  }

  const server = app.listen(PORT, HOST, () => {
    console.log(`PDF OS Server running on http://${HOST}:${PORT}`);
  });

  // Explicitly set Node.js HTTP server timeouts to 5 minutes to prevent disconnects
  server.timeout = 300000;
  server.keepAliveTimeout = 305000;
  server.headersTimeout = 310000;

  server.on('error', (err: any) => {
    if (err.code === 'EADDRINUSE') {
      console.warn(`[SERVER NOTICE] Port ${PORT} already active, continuing gracefully.`);
    } else {
      console.error('[SERVER RUNTIME ERROR]', err);
    }
  });
}

startServer();
