const { chromium } = require('D:/Hermes/imported_sources/playwright-mcp/node_modules/playwright');
const target = process.argv[2] || 'http://127.0.0.1:7777/';
const parsed = new URL(target);
const allowed = parsed.protocol === 'file:' || (parsed.protocol === 'http:' && (parsed.hostname === '127.0.0.1' || parsed.hostname === 'localhost'));
if (!allowed) { console.error(JSON.stringify({ok:false,error:'external_urls_blocked',local_only:true})); process.exit(2); }
(async()=>{
  const browser = await chromium.launch({headless:true});
  try {
    const page = await browser.newPage();
    await page.goto(target, {waitUntil:'domcontentloaded', timeout:20000});
    console.log(JSON.stringify({ok:true,url:page.url(),title:await page.title(),local_only:true}));
  } finally { await browser.close(); }
})().catch(err=>{ console.error(JSON.stringify({ok:false,error:String(err),local_only:true})); process.exit(1); });
