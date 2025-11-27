import streamlit as st
import streamlit.components.v1 as components
import pdfplumber
import json
import re
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
from io import BytesIO
import os

# ==================== CONFIG ====================
# ❗ Replace this with your actual key or use an environment variable
OPENROUTER_API_KEY = "sk-or-v1"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Different models for different stages
MODEL_STAGE2 = "x-ai/grok-4-fast"  # Best for PDF parsing
MODEL_STAGE3 = "meta-llama/llama-3.3-70b-instruct:free"  # Best for categorization

MAX_WORKERS = 8
CHUNK_SIZE_STAGE2 = 3500
CHUNK_SIZE_STAGE3 = 100

# Path to the HTML redaction tool
HTML_REDACTION_PATH = r"C:\Users\rajro\majorproject\pdf editor.html"

# ==================== SESSION STATE MANAGEMENT ====================
if 'redacted_pdf' not in st.session_state:
    st.session_state.redacted_pdf = None
if 'stage2_output' not in st.session_state:
    st.session_state.stage2_output = None
if 'stage3_output' not in st.session_state:
    st.session_state.stage3_output = None

# ==================== HELPER CLASSES ====================
class ProgressTracker:
    def __init__(self, total):
        self.lock = threading.Lock()
        self.completed = 0
        self.total = total

    def increment(self):
        with self.lock:
            self.completed += 1
            return self.completed

# ==================== PRIVACY PROTECTION ====================
def redact_sensitive_info(text):
    """
    Redacts personal information from text before sending to LLM.
    Protects: Account numbers, PAN, Aadhaar, IFSC, customer names, addresses.
    """
    # Redact account numbers (various formats)
    text = re.sub(r'\b\d{9,18}\b', '[ACCOUNT_REDACTED]', text)

    # Redact PAN card (e.g., ABCDE1234F)
    text = re.sub(r'\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b', '[PAN_REDACTED]', text)

    # Redact Aadhaar (12 digits, with or without spaces/dashes)
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[AADHAAR_REDACTED]', text)

    # Redact IFSC codes (e.g., SBIN0001234)
    text = re.sub(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', '[IFSC_REDACTED]', text)

    # Redact email addresses
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        '[EMAIL_REDACTED]',
        text,
    )

    # Redact phone numbers (10 digits with optional +91 prefix)
    text = re.sub(
        r'\+?91[\s-]?\d{10}|\b\d{10}\b',
        '[PHONE_REDACTED]',
        text,
    )

    # Redact common personal info headers (case insensitive)
    patterns_to_remove = [
        r'Customer Name\s*:?\s*[A-Za-z\s]+',
        r'Account Holder\s*:?\s*[A-Za-z\s]+',
        r'Name\s*:?\s*[A-Z][a-z]+\s+[A-Z][a-z]+',
        r'Address\s*:?\s*.+',
        r'Date of Birth\s*:?\s*\d{2}[-/]\d{2}[-/]\d{4}',
        r'DOB\s*:?\s*\d{2}[-/]\d{2}[-/]\d{4}',
    ]

    for pattern in patterns_to_remove:
        text = re.sub(pattern, '[PERSONAL_INFO_REDACTED]', text, flags=re.IGNORECASE)

    return text


def extract_transaction_tables(pdf_file):
    """
    Extract only transaction table regions from PDF, avoiding headers/footers with personal info.
    """
    transaction_data = []

    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Try to extract tables first (more structured)
            tables = page.extract_tables()

            if tables:
                for table in tables:
                    # Filter out header rows and empty rows
                    for row in table:
                        if row and any(row):  # Non-empty row
                            row_text = ' '.join([str(cell) for cell in row if cell])

                            # Check if row looks like a transaction (has date and amount patterns)
                            if (
                                re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', row_text)
                                and re.search(r'\d+[.,]\d{2}', row_text)
                            ):
                                transaction_data.append(row_text)
            else:
                # Fallback: extract text and filter transaction lines
                page_text = page.extract_text()
                if page_text:
                    lines = page_text.split('\n')
                    for line in lines:
                        # Only include lines that look like transactions
                        if (
                            re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', line)
                            and re.search(r'\d+[.,]\d{2}', line)
                        ):
                            transaction_data.append(line)

    return '\n'.join(transaction_data)


def extract_text_with_privacy(pdf_file):
    """
    Enhanced extraction: Extract transaction tables + apply privacy redaction.
    """
    # Step 1: Extract only transaction table regions
    transaction_text = extract_transaction_tables(pdf_file)

    # Step 2: Apply additional privacy redaction
    cleaned_text = redact_sensitive_info(transaction_text)

    return cleaned_text

# ==================== STAGE 2: PDF PARSING (PRIVACY-ENHANCED) ====================
def chunk_text_smart(text, max_len=CHUNK_SIZE_STAGE2):
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > max_len and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_len = line_len
        else:
            current_chunk.append(line)
            current_len += line_len

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks


def call_llm_stage2(chunk, chunk_index, total_chunks):
    prompt = f"""Extract bank transactions from this statement text. Return JSON array only.

RULES:
1. Extract: date (DD-MM-YYYY), description, debit, credit, balance
2. If date missing, use previous transaction date or infer from context
3. Debit OR credit per transaction (not both)
4. Set debit/credit to 0 if not applicable
5. Output ONLY valid JSON array - no markdown, no explanations

CHUNK {chunk_index + 1}/{total_chunks}:
{chunk}

JSON OUTPUT:
[{{"date":"DD-MM-YYYY","description":"text","debit":0,"credit":0,"balance":0}}]"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_STAGE2,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise bank statement parser. Output ONLY valid JSON arrays with accurate transaction data. No markdown, no text, just JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=8000,
            temperature=0,
        )

        response = completion.choices[0].message.content.strip()

        # Clean response
        if response.startswith('```json'):
            response = response.replace('```json', '').replace('```', '').strip()
        if response.startswith('```'):
            response = response.replace('```', '').strip()

        if not response.startswith('['):
            response = '[' + response
        if not response.endswith(']'):
            last_brace = response.rfind('}')
            if last_brace != -1:
                response = response[: last_brace + 1] + ']'
            else:
                response += ']'

        transactions = json.loads(response)

        if not isinstance(transactions, list):
            return {
                "chunk_index": chunk_index,
                "transactions": [],
                "error": "Invalid format",
            }

        validated = []
        for txn in transactions:
            if isinstance(txn, dict):
                validated.append(
                    {
                        "date": txn.get("date", ""),
                        "description": txn.get("description", ""),
                        "debit": float(txn.get("debit", 0)),
                        "credit": float(txn.get("credit", 0)),
                        "balance": float(txn.get("balance", 0)),
                    }
                )

        return {"chunk_index": chunk_index, "transactions": validated, "error": None}

    except Exception as e:
        return {
            "chunk_index": chunk_index,
            "transactions": [],
            "error": str(e)[:100],
        }


def parse_pdf_stage2(file, max_workers=MAX_WORKERS):
    start_time = datetime.now()

    with st.spinner("🔒 Extracting transaction data with privacy protection..."):
        # Use privacy-enhanced extraction
        raw_text = extract_text_with_privacy(file)

    if not raw_text.strip():
        st.error("❌ No transaction data extracted from PDF. Please check if PDF is readable.")
        return None

    st.success(f"✅ Extracted {len(raw_text)} characters (privacy-protected)")

    # Show info about privacy protection
    with st.expander("🔒 Privacy Protection Applied"):
        st.markdown(
            """
        **The following information has been redacted before sending to LLM:**
        - ✅ Account numbers
        - ✅ PAN card numbers
        - ✅ Aadhaar numbers
        - ✅ IFSC codes
        - ✅ Email addresses
        - ✅ Phone numbers
        - ✅ Personal info headers (Name, Address, DOB)
        - ✅ Non-transaction text (headers/footers)
        
        **Only transaction data is sent to the LLM for processing.**
        """
        )

    chunks = chunk_text_smart(raw_text)
    total_chunks = len(chunks)

    st.info(f"📦 Processing {total_chunks} chunks with {max_workers} parallel workers")

    progress_tracker = ProgressTracker(total_chunks)
    progress_bar = st.progress(0)
    status_text = st.empty()

    chunk_results = [None] * total_chunks
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(call_llm_stage2, chunk, idx, total_chunks): idx
            for idx, chunk in enumerate(chunks)
        }

        for future in as_completed(future_to_chunk):
            result = future.result()
            chunk_idx = result['chunk_index']

            if result['error']:
                errors.append(f"Chunk {chunk_idx + 1}: {result['error']}")

            chunk_results[chunk_idx] = result['transactions']

            completed = progress_tracker.increment()
            progress = completed / total_chunks
            progress_bar.progress(progress)
            status_text.text(f"⚡ Stage 2: {completed}/{total_chunks} chunks processed...")

    all_transactions = []
    for chunk_txns in chunk_results:
        if chunk_txns:
            all_transactions.extend(chunk_txns)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    status_text.empty()
    progress_bar.empty()

    if errors:
        with st.expander("⚠️ View Stage 2 Warnings"):
            for error in errors:
                st.write(f"- {error}")

    st.success(
        f"✅ Stage 2 Complete: Parsed {len(all_transactions)} transactions in {duration:.2f}s (Privacy Protected)"
    )

    return all_transactions

# ==================== STAGE 3: RULE-BASED + AI CATEGORIZATION ====================
def extract_payee(desc: str) -> str:
    desc = str(desc)

    match = re.search(r'\(([^)]+)\)', desc)
    if match:
        name = match.group(1).strip()
        if 1 < len(name) < 40:
            return name

    match = re.search(r'@[\w]+[\W]*([A-Za-z][A-Za-z\s]{1,30})', desc)
    if match:
        return match.group(1).strip()

    match = re.search(
        r'(?:to|Paymen(?:t)?|deposit|Credited to)\s*([A-Za-z][A-Za-z\s]{1,40})',
        desc,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    match = re.search(r'P2A/\d+/([A-Za-z\s\.]+)/', desc)
    if match:
        return match.group(1).strip()

    match = re.search(r':([a-z0-9\.\-]+@[a-z]+)', desc, re.IGNORECASE)
    if match:
        return match.group(1)

    words = re.findall(r'[A-Za-z]{2,}', desc)
    if len(words) > 0:
        return ' '.join(words[:3]).strip()
    return 'Unknown'


def get_mode(desc: str) -> str:
    desc_upper = desc.upper()
    if 'UPI' in desc_upper:
        return 'UPI'
    elif 'NEFT' in desc_upper:
        return 'NEFT'
    elif 'RTGS' in desc_upper:
        return 'RTGS'
    elif 'IMPS' in desc_upper:
        return 'IMPS'
    elif 'ATM' in desc_upper:
        return 'ATM-CASH'
    elif 'ACH' in desc_upper or 'ECS' in desc_upper:
        return 'ACH'
    elif 'STAT' in desc_upper:
        return 'STAT'
    else:
        return 'Others'


def process_stage3_rules(transactions):
    df = pd.DataFrame(transactions)

    df['description'] = df['description'].fillna('').astype(str)
    df['mode_of_transaction'] = df['description'].apply(get_mode)
    df['paid_to'] = df['description'].apply(extract_payee)

    df['date'] = df['date'].astype(str).str.strip()
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df['year'] = df['date_parsed'].dt.year.fillna('').astype(str)

    return df


def categorize_chunk_stage3(transactions_data, chunk_index):
    lightweight_data = [
        {
            "index": i,
            "paid_to": str(t["paid_to"]),
            "description": str(t["description"]),
        }
        for i, t in enumerate(transactions_data)
    ]

    input_str = json.dumps(lightweight_data, indent=2)

    prompt = f"""You are an expert Indian banking transaction classifier. Use the examples below to classify transactions accurately.

CATEGORY DEFINITIONS WITH INDIAN EXAMPLES:

1. **Food & Dining**
   Examples: Swiggy, Zomato, Blinkit, Zepto, Dunzo, BigBasket, McDonald's, KFC, Dominos, Pizza Hut, Starbucks, Cafe Coffee Day, Chaayos, Haldiram's, MTR Foods, Saravana Bhavan, Paradise Biryani, Barbeque Nation

2. **Recharge & Bills**
   Examples: Airtel, Jio, Vi/Vodafone Idea, BSNL, Tata Sky, Dish TV, Airtel Digital TV, ACT Fibernet, Hathway, JioFiber, BESCOM, MSEDCL, BSES, Torrent Power, Adani Gas

3. **Entertainment**
   Examples: Netflix, Amazon Prime Video, Disney+ Hotstar, Sony Liv, Zee5, BookMyShow, PVR Cinemas, INOX, Spotify, YouTube Premium, Gaana, JioSaavn, PlayStation, Xbox, Steam

4. **Investment & Trading**
   Examples: Zerodha, Groww, Upstox, Angel One, 5Paisa, ICICI Direct, HDFC Securities, Kotak Securities, Paytm Money, ET Money, Coin by Zerodha, Scripbox, Kuvera, INDmoney, Smallcase, Mutual Fund, SIP

5. **Shopping & E-commerce**
   Examples: Amazon, Flipkart, Myntra, Ajio, Meesho, Nykaa, FirstCry, Reliance Digital, Croma, Decathlon, Westside, Lifestyle, Shoppers Stop, H&M, Zara, Nike, Adidas, Lenskart

6. **Transfer (P2P)**
   Examples: Person names like "Rahul Sharma", "Priya Patel", "Amit Kumar", UPI transfers to individuals, split bills, friend payments

7. **Bills & Utilities**
   Examples: Credit card payments (HDFC, ICICI, SBI, Axis), Loan EMI, Insurance (LIC, Max Life, HDFC Life), Property Tax, Society Maintenance, Water Bill

8. **Salary & Income**
   Examples: Salary credit, freelance payment, consulting fee, bonus, incentive, commission

9. **ATM Withdrawal**
   Examples: ATM cash withdrawal from any bank ATM

10. **Transport & Travel**
    Examples: Uber, Ola, Rapido, Bounce, Yulu, IRCTC, MakeMyTrip, Goibibo, Cleartrip, SpiceJet, IndiGo, Air India, Indian Oil, HP Petrol, BPCL, Fastag

11. **Healthcare & Fitness**
    Examples: Apollo Pharmacy, 1mg, PharmEasy, Netmeds, Cult.fit, HealthifyMe, Practo, Apollo Hospital, Fortis, Max Healthcare

12. **Education**
    Examples: Byju's, Unacademy, Vedantu, Coursera, Udemy, Khan Academy, school fees, college fees, coaching classes

13. **Others**
    Everything else that doesn't fit the above categories

FEW-SHOT EXAMPLES:
- {{"paid_to": "Swiggy", "description": "UPI/Swiggy/Food Order"}} → Food & Dining
- {{"paid_to": "Zerodha", "description": "Mutual Fund SIP"}} → Investment & Trading
- {{"paid_to": "Airtel", "description": "Mobile Recharge Prepaid"}} → Recharge & Bills
- {{"paid_to": "BookMyShow", "description": "Movie Ticket PVR"}} → Entertainment
- {{"paid_to": "Amazon", "description": "Shopping Electronics"}} → Shopping & E-commerce
- {{"paid_to": "Rahul Kumar", "description": "UPI/P2A/Split Bill"}} → Transfer (P2P)
- {{"paid_to": "HDFC Bank", "description": "Credit Card Bill Payment"}} → Bills & Utilities
- {{"paid_to": "Company Name", "description": "Salary for October 2024"}} → Salary & Income
- {{"paid_to": "ATM", "description": "ATM Cash Withdrawal HDFC"}} → ATM Withdrawal
- {{"paid_to": "Uber", "description": "Ride from Home to Office"}} → Transport & Travel
- {{"paid_to": "Apollo Pharmacy", "description": "Medicine Purchase"}} → Healthcare & Fitness
- {{"paid_to": "Byju's", "description": "Course Subscription"}} → Education

INPUT DATA (Chunk {chunk_index + 1}):
{input_str}

INSTRUCTIONS:
1. Match "paid_to" and "description" against the examples above
2. Classify into ONE of these categories: Food & Dining, Recharge & Bills, Entertainment, Investment & Trading, Shopping & E-commerce, Transfer (P2P), Bills & Utilities, Salary & Income, ATM Withdrawal, Transport & Travel, Healthcare & Fitness, Education, Others
3. Return ONLY a JSON array with index and category

OUTPUT FORMAT:
[
  {{"index": 0, "category": "Food & Dining"}},
  {{"index": 1, "category": "Investment & Trading"}}
]

CRITICAL: Output ONLY valid JSON array, no markdown, no explanations."""

    try:
        completion = client.chat.completions.create(
            model=MODEL_STAGE3,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise transaction classifier. Output ONLY JSON arrays.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=4000,
        )

        response = completion.choices[0].message.content.strip()

        if response.startswith('```json'):
            response = response.replace('```json', '').replace('```', '').strip()
        if response.startswith('```'):
            response = response.replace('```', '').strip()

        if not response.startswith('['):
            response = '[' + response
        if not response.endswith(']'):
            response += ']'

        categories = json.loads(response)
        return {"chunk_index": chunk_index, "categories": categories, "error": None}

    except Exception as e:
        return {"chunk_index": chunk_index, "categories": None, "error": str(e)}


def process_stage3_complete(transactions, max_workers=5):
    start_time = datetime.now()

    with st.spinner("🔧 Applying rule-based processing..."):
        df = process_stage3_rules(transactions)

    st.success("✅ Rule-based processing complete")

    total_chunks = (len(df) + CHUNK_SIZE_STAGE3 - 1) // CHUNK_SIZE_STAGE3
    st.info(f"🤖 Starting AI categorization: {total_chunks} chunks with {max_workers} workers")

    progress_tracker = ProgressTracker(total_chunks)
    progress_bar = st.progress(0)
    status_text = st.empty()

    all_categories = ['Others'] * len(df)

    chunks = []
    for i in range(total_chunks):
        start_idx = i * CHUNK_SIZE_STAGE3
        end_idx = min((i + 1) * CHUNK_SIZE_STAGE3, len(df))

        chunk_data = df.iloc[start_idx:end_idx][['description', 'paid_to']].to_dict(
            'records'
        )
        chunks.append((chunk_data, i))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(categorize_chunk_stage3, data, idx): idx
            for data, idx in chunks
        }

        for future in as_completed(future_to_chunk):
            result = future.result()

            if not result['error'] and result['categories']:
                chunk_start = result['chunk_index'] * CHUNK_SIZE_STAGE3
                for item in result['categories']:
                    original_idx = chunk_start + item['index']
                    if original_idx < len(all_categories):
                        all_categories[original_idx] = item['category']

            completed = progress_tracker.increment()
            progress = completed / total_chunks
            progress_bar.progress(progress)
            status_text.text(
                f"⚡ Stage 3: {completed}/{total_chunks} chunks categorized..."
            )

    df['category'] = all_categories

    output_columns = [
        'date',
        'year',
        'description',
        'mode_of_transaction',
        'paid_to',
        'debit',
        'credit',
        'balance',
        'category',
    ]
    df_final = df[output_columns].copy()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    status_text.empty()
    progress_bar.empty()

    st.success(
        f"✅ Stage 3 Complete: Categorized {len(df_final)} transactions in {duration:.2f}s"
    )

    return df_final.to_dict('records')

# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="3-Step Banking Pipeline", layout="wide")
st.title("🏦 3-Step Banking Transaction Pipeline")
st.markdown("**Step 1: Redact PDF → Step 2: Parse → Step 3: Categorize**")

# Tab-based navigation
tab1, tab2, tab3 = st.tabs(
    ["🔒 Step 1: Redact PDF", "📄 Step 2: Parse PDF", "🎯 Step 3: Categorize"]
)

# ==================== STEP 1 TAB: PDF REDACTION ====================
with tab1:
    st.header("🔒 Step 1: Redact Sensitive Information")
    st.write("Use the PDF editor to hide/redact sensitive information before processing")

    # Check if HTML file exists
    if os.path.exists(HTML_REDACTION_PATH):
        # Read and display the HTML file
        with open(HTML_REDACTION_PATH, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Display the HTML in an iframe
        components.html(html_content, height=800, scrolling=True)

        st.info("💡 **Instructions:**")
        st.markdown(
            """
        1. Load your PDF in the editor above  
        2. Use the redaction tools to hide sensitive information (account numbers, names, etc.)  
        3. Download the redacted PDF  
        4. Upload the redacted PDF in **Step 2: Parse PDF** tab
        """
        )

    else:
        st.error(f"❌ HTML file not found at: {HTML_REDACTION_PATH}")
        st.info("Please make sure the PDF editor HTML file exists at the specified path.")

        # Allow manual upload of redacted PDF
        st.markdown("---")
        st.subheader("📤 Or Upload Already Redacted PDF")
        redacted_upload = st.file_uploader(
            "Upload Redacted PDF", type="pdf", key="redacted_pdf_upload"
        )

        if redacted_upload:
            st.session_state.redacted_pdf = redacted_upload
            st.success("✅ Redacted PDF uploaded! Go to Step 2 to parse it.")

# ==================== STEP 2 TAB: PDF PARSING ====================
with tab2:
    st.header("📄 Step 2: Parse PDF Transactions")
    st.write("Extract raw transaction data from your (redacted) bank statement")

    # Privacy info banner
    st.info("🔒 **Privacy Protected**: Personal information is automatically redacted before sending to LLM")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Bank Statement PDF", type="pdf", key="pdf_upload"
        )
    with col2:
        workers_stage2 = st.slider("Workers", 1, 10, 8, key="workers2")

    # Show if redacted PDF is available from Step 1
    if st.session_state.redacted_pdf:
        st.info("✅ Redacted PDF available from Step 1")

    if uploaded_file:
        if st.button("🚀 Start Step 2: Parse PDF", type="primary", key="start_stage2"):
            uploaded_file.seek(0)
            result = parse_pdf_stage2(uploaded_file, max_workers=workers_stage2)

            if result:
                st.session_state.stage2_output = result

                # Show metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Transactions", len(result))
                with col2:
                    total_debit = sum(t['debit'] for t in result)
                    st.metric("Total Debits", f"₹{total_debit:,.2f}")
                with col3:
                    total_credit = sum(t['credit'] for t in result)
                    st.metric("Total Credits", f"₹{total_credit:,.2f}")

                # Show sample
                st.subheader("📊 Sample Output (First 5)")
                st.json(result[:5])

                st.success("✅ Step 2 Complete! Data ready for Step 3")
                st.info("👉 Go to **Step 3: Categorize** tab to continue")

    # Show existing Stage 2 output if available
    if st.session_state.stage2_output:
        st.markdown("---")
        st.subheader("💾 Existing Step 2 Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Transactions", len(st.session_state.stage2_output))
        with col2:
            total_debit = sum(t['debit'] for t in st.session_state.stage2_output)
            st.metric("Total Debits", f"₹{total_debit:,.2f}")
        with col3:
            total_credit = sum(t['credit'] for t in st.session_state.stage2_output)
            st.metric("Total Credits", f"₹{total_credit:,.2f}")

        # Download Stage 2 JSON
        json_str = json.dumps(st.session_state.stage2_output, indent=2)
        st.download_button(
            label="📥 Download Step 2 JSON (Optional)",
            data=json_str,
            file_name=f"stage2_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_stage2_json",
        )

# ==================== STEP 3 TAB: CATEGORIZATION ====================
with tab3:
    st.header("🎯 Step 3: Categorize Transactions")
    st.write("Apply rule-based + AI categorization to your parsed transactions.")

    workers_stage3 = st.slider("Workers", 1, 10, 5, key="workers3")

    st.subheader("📥 Choose Input for Categorization")
    source = st.radio(
        "Select input source",
        ["Use Step 2 data", "Upload JSON"],
        key="stage3_source",
    )

    transactions_input = None

    if source == "Use Step 2 data":
        if st.session_state.stage2_output:
            st.success(
                f"Using {len(st.session_state.stage2_output)} transactions from Step 2."
            )
            transactions_input = st.session_state.stage2_output
        else:
            st.warning("No Step 2 data available. Please complete Step 2 or upload JSON.")
    else:
        json_file = st.file_uploader(
            "Upload transactions JSON (array of objects with date, description, debit, credit, balance)",
            type=["json"],
            key="stage3_json_upload",
        )
        if json_file:
            try:
                data_bytes = json_file.read()
                transactions_input = json.loads(data_bytes.decode("utf-8"))
                if not isinstance(transactions_input, list):
                    st.error("❌ JSON must be an array of transaction objects.")
                    transactions_input = None
                else:
                    st.success(f"Loaded {len(transactions_input)} transactions from JSON.")
            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")
                transactions_input = None

    if transactions_input and st.button(
        "🚀 Start Step 3: Categorize", type="primary", key="start_stage3"
    ):
        result_stage3 = process_stage3_complete(
            transactions_input, max_workers=workers_stage3
        )
        st.session_state.stage3_output = result_stage3

        df_final = pd.DataFrame(result_stage3)

        st.subheader("📊 Category Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Transactions", len(df_final))
        with col2:
            st.metric("Unique Categories", df_final['category'].nunique())

        st.subheader("📈 Spend by Category (Debits)")
        spend_by_cat = (
            df_final.groupby('category')['debit'].sum().sort_values(ascending=False)
        )
        if len(spend_by_cat) > 0:
            st.bar_chart(spend_by_cat)

        st.subheader("📋 Sample Categorized Transactions (First 10)")
        st.dataframe(df_final.head(10))

        # Download options
        csv_data = df_final.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Categorized CSV",
            data=csv_data,
            file_name=f"categorized_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_stage3_csv",
        )

        # Optional Excel download
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df_final.to_excel(writer, index=False, sheet_name="Transactions")
        excel_buffer.seek(0)
        st.download_button(
            label="📥 Download Categorized Excel",
            data=excel_buffer,
            file_name=f"categorized_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_stage3_excel",
        )

    # Show existing Stage 3 data if present
    if st.session_state.stage3_output:
        st.markdown("---")
        st.subheader("💾 Existing Step 3 Data")
        df_existing = pd.DataFrame(st.session_state.stage3_output)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Transactions", len(df_existing))
        with col2:
            st.metric("Unique Categories", df_existing['category'].nunique())

        st.dataframe(df_existing.head(10))

        csv_existing = df_existing.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Existing Categorized CSV",
            data=csv_existing,
            file_name="existing_categorized_transactions.csv",
            mime="text/csv",
            key="download_existing_stage3_csv",
        )
