
# 13 LLM 在垂直领域的应用

## 13.1 金融领域

### 13.1.1 智能投顾系统

LLM可以用于构建智能投资顾问系统，为用户提供个性化的投资建议。

```python
import openai
import yfinance as yf

class InvestmentAdvisor:
    def __init__(self, api_key):
        openai.api_key = api_key

    def get_stock_data(self, ticker):
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info['longName'],
            "sector": info['sector'],
            "price": info['currentPrice'],
            "pe_ratio": info['forwardPE'],
            "dividend_yield": info.get('dividendYield', 0) * 100
        }

    def generate_advice(self, user_profile, stock_data):
        prompt = f"""
        Given the following user profile and stock information, provide a brief investment advice:

        User Profile:
        - Risk tolerance: {user_profile['risk_tolerance']}
        - Investment horizon: {user_profile['investment_horizon']}
        - Investment goals: {user_profile['investment_goals']}

        Stock Information:
        - Name: {stock_data['name']}
        - Sector: {stock_data['sector']}
        - Current Price: ${stock_data['price']}
        - P/E Ratio: {stock_data['pe_ratio']}
        - Dividend Yield: {stock_data['dividend_yield']}%

        Investment Advice:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

# 使用示例
advisor = InvestmentAdvisor("your-openai-api-key")

user_profile = {
    "risk_tolerance": "moderate",
    "investment_horizon": "5-10 years",
    "investment_goals": "long-term growth and some income"
}

stock_data = advisor.get_stock_data("AAPL")
advice = advisor.generate_advice(user_profile, stock_data)

print(f"Investment Advice for {stock_data['name']}:")
print(advice)
```

### 13.1.2 风险评估模型

LLM可以辅助构建更全面的风险评估模型，分析各种数据源以评估金融风险。

```python
import openai
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class RiskAssessmentModel:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.scaler = StandardScaler()

    def preprocess_data(self, financial_data):
        # 假设financial_data是一个包含各种财务指标的DataFrame
        numeric_data = financial_data.select_dtypes(include=[np.number])
        scaled_data = self.scaler.fit_transform(numeric_data)
        return pd.DataFrame(scaled_data, columns=numeric_data.columns)

    def calculate_risk_score(self, preprocessed_data):
        # 使用简单的加权平均来计算风险分数
        weights = {
            'debt_to_equity': 0.3,
            'current_ratio': -0.2,
            'return_on_equity': -0.2,
            'beta': 0.3
        }
        risk_score = sum(preprocessed_data[col] * weight for col, weight in weights.items() if col in preprocessed_data)
        return risk_score.iloc[0]  # 假设我们只处理一个公司的数据

    def interpret_risk(self, risk_score, company_info):
        prompt = f"""
        Given the following information about a company and its calculated risk score, provide a detailed risk assessment:

        Company: {company_info['name']}
        Sector: {company_info['sector']}
        Risk Score: {risk_score:.2f} (on a scale where higher scores indicate higher risk)

        Key Financial Metrics:
        - Debt to Equity Ratio: {company_info['debt_to_equity']:.2f}
        - Current Ratio: {company_info['current_ratio']:.2f}
        - Return on Equity: {company_info['return_on_equity']:.2f}%
        - Beta: {company_info['beta']:.2f}

        Please provide:
        1. An overall risk assessment
        2. Key risk factors
        3. Potential mitigating factors
        4. Recommendations for risk management

        Risk Assessment:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

# 使用示例
risk_model = RiskAssessmentModel("your-openai-api-key")

# 假设这是从某个数据源获取的财务数据
financial_data = pd.DataFrame({
    'debt_to_equity': [1.5],
    'current_ratio': [1.2],
    'return_on_equity': [15.0],
    'beta': [1.3]
})

company_info = {
    'name': 'TechCorp Inc.',
    'sector': 'Technology',
    'debt_to_equity': 1.5,
    'current_ratio': 1.2,
    'return_on_equity': 15.0,
    'beta': 1.3
}

preprocessed_data = risk_model.preprocess_data(financial_data)
risk_score = risk_model.calculate_risk_score(preprocessed_data)
risk_assessment = risk_model.interpret_risk(risk_score, company_info)

print(f"Risk Assessment for {company_info['name']}:")
print(risk_assessment)
```

### 13.1.3 金融报告自动生成

LLM可以用于自动生成金融报告，总结复杂的财务数据和市场趋势。

```python
import openai
import yfinance as yf
from datetime import datetime, timedelta

class FinancialReportGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key

    def get_stock_data(self, ticker, period="1mo"):
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return {
            "name": stock.info['longName'],
            "sector": stock.info['sector'],
            "current_price": hist['Close'][-1],
            "price_change": (hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0] * 100,
            "volume": hist['Volume'].mean(),
            "high": hist['High'].max(),
            "low": hist['Low'].min()
        }

    def get_financial_metrics(self, ticker):
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "market_cap": info.get('marketCap', 'N/A'),
            "pe_ratio": info.get('forwardPE', 'N/A'),
            "dividend_yield": info.get('dividendYield', 'N/A'),
            "eps": info.get('trailingEps', 'N/A'),
            "revenue": info.get('totalRevenue', 'N/A'),
            "profit_margin": info.get('profitMargins', 'N/A')
        }

    def generate_report(self, ticker):
        stock_data = self.get_stock_data(ticker)
        financial_metrics = self.get_financial_metrics(ticker)
        
        prompt = f"""
        Generate a comprehensive financial report for {stock_data['name']} ({ticker}) based on the following data:

        Stock Performance (Last Month):
        - Current Price: ${stock_data['current_price']:.2f}
        - Price Change: {stock_data['price_change']:.2f}%
        - Average Daily Volume: {stock_data['volume']:,.0f}
        - Highest Price: ${stock_data['high']:.2f}
        - Lowest Price: ${stock_data['low']:.2f}

        Financial Metrics:
        - Market Cap: ${financial_metrics['market_cap']:,.0f}
        - P/E Ratio: {financial_metrics['pe_ratio']}
        - Dividend Yield: {financial_metrics['dividend_yield']:.2%}
        - Earnings Per Share: ${financial_metrics['eps']:.2f}
        - Revenue: ${financial_metrics['revenue']:,.0f}
        - Profit Margin: {financial_metrics['profit_margin']:.2%}

        Please provide:
        1. An executive summary
        2. Analysis of stock performance
        3. Evaluation of financial health
        4. Industry comparison
        5. Future outlook and recommendations

        Financial Report:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

# 使用示例
report_generator = FinancialReportGenerator("your-openai-api-key")
ticker = "AAPL"
report = report_generator.generate_report(ticker)

print(f"Financial Report for {ticker}:")
print(report)
```

## 13.2 法律领域

### 13.2.1 法律文书智能审核

LLM可以用于审核法律文书，检查其完整性、一致性和潜在问题。

```python
import openai

class LegalDocumentReviewer:
    def __init__(self, api_key):
        openai.api_key = api_key

    def review_document(self, document_text, document_type):
        prompt = f"""
        Review the following {document_type} for potential issues, inconsistencies, or areas that need improvement:

        {document_text}

        Please provide:
        1. Overall assessment
        2. Potential legal issues or risks
        3. Inconsistencies or ambiguities
        4. Suggestions for improvement
        5. Compliance with relevant laws and regulations

        Document Review:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

# 使用示例
reviewer = LegalDocumentReviewer("your-openai-api-key")

contract_text = """
This Agreement is made on [DATE] between [PARTY A] and [PARTY B].

1. Services: [PARTY A] agrees to provide [SERVICES] to [PARTY B].
2. Payment: [PARTY B] shall pay [AMOUNT] for the services within [TIMEFRAME].
3. Term: This Agreement shall commence on [START DATE] and continue until [END DATE].
4. Termination: Either party may terminate this Agreement with [NOTICE PERIOD] written notice.
5. Confidentiality: Both parties agree to keep all information confidential for [DURATION].
6. Governing Law: This Agreement shall be governed by the laws of [JURISDICTION].

Signed by:
[PARTY A]                    [PARTY B]
"""

review = reviewer.review_document(contract_text, "Service Agreement")
print("Legal Document Review:")
print(review)
```

### 13.2.2 案例检索与分析

LLM可以辅助法律专业人士进行案例检索和分析，提高工作效率。

```python
import openai

class LegalCaseAnalyzer:
    def __init__(self, api_key):
        openai.api_key = api_key

    def search_relevant_cases(self, query, num_results=5):
        # 这里应该是一个实际的案例数据库搜索功能
        # 为了示例，我们只返回一些模拟的案例标题
        return [
            "Smith v. Jones (2019): Breach of Contract",
            "Brown v. State (2020): Criminal Procedure",
            "Johnson v. City (2018): Civil Rights",
            "Davis v. Corporation (2021): Employment Law",
            "Wilson v. Agency (2017): Administrative Law"
        ][:num_results]

    def analyze_case(self, case_title, case_summary):
        prompt = f"""
        Analyze the following legal case:

        Case: {case_title}
        Summary: {case_summary}

        Please provide:
        1. Key legal issues
        2. Relevant legal principles
        3. Court's reasoning
        4. Implications of the decision
        5. Potential applications to similar cases

        Case Analysis:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=400,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

    def compare_cases(self, case1, case2):
        prompt = f"""
        Compare and contrast the following two legal cases:

        Case 1: {case1}
        Case 2: {case2}

        Please provide:
        1. Similarities in legal issues
        2. Differences in court decisions
        3. Evolution of legal principles (if applicable)
        4. Implications for future cases

        Case Comparison:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=400,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

# 使用示例
analyzer = LegalCaseAnalyzer("your-openai-api-key")

query = "breach of contract in software development"
relevant_cases = analyzer.search_relevant_cases(query)
print("Relevant Cases:")
for case in relevant_cases:
    print(f"- {case}")

case_summary = """
The plaintiff, a software company, sued the defendant, a client, for breach of contract. 
The defendant had terminated the contract early, claiming that the software did not meet specified requirements. 
The court found in favor of the plaintiff, ruling that the defendant had not provided sufficient opportunity for the plaintiff to address the alleged deficiencies as stipulated in the contract.
"""

analysis = analyzer.analyze_case(relevant_cases[0], case_summary)
print("\nCase Analysis:")
print(analysis)

comparison = analyzer.compare_cases(relevant_cases[0], relevant_cases[1])
print("\nCase Comparison:")
print(comparison)
```

### 13.2.3 合同自动生成与审核

LLM可以用于自动生成合同草稿，并对现有合同进行审核。

```python
import openai

class ContractGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_contract(self, contract_type, parties, key_terms):
        prompt = f"""
        Generate a {contract_type} contract between {parties['party_a']} and {parties['party_b']} with the following key terms:

        {key_terms}

        The contract should include standard clauses such as:
        1. Definitions
        2. Term and Termination
        3. Payment Terms
        4. Confidentiality
        5. Intellectual Property
        6. Liability and Indemnification
        7. Governing Law and Jurisdiction
        8. Entire Agreement

        Contract:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

    def review_contract(self, contract_text):
        prompt = f"""
        Review the following contract for potential issues, missing clauses, or areas that need improvement:

        {contract_text}

        Please provide:
        1. Overall assessment
        2. Potential legal issues or risks
        3. Missing or inadequate clauses
        4. Suggestions for improvement
        5. Compliance with relevant laws and regulations

        Contract Review:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

# 使用示例
contract_tool = ContractGenerator("your-openai-api-key")

contract_type = "Software Development Agreement"
parties = {
    "party_a": "TechCorp Inc.",
    "party_b": "Client LLC"
}
key_terms = """
1. TechCorp Inc. will develop a custom CRM software for Client LLC.
2. The project will be completed in 6 months from the contract start date.
3. The total cost of the project is $100,000, payable in monthly installments.
4. TechCorp Inc. will provide 12 months of support and maintenance after project completion.
5. Client LLC will own the intellectual property rights to the custom software.
"""

generated_contract = contract_tool.generate_contract(contract_type, parties, key_terms)
print("Generated Contract:")
print(generated_contract)

review = contract_tool.review_contract(generated_contract)
print("\nContract Review:")
print(review)
```

## 13.3 医疗健康领域

### 13.3.1 医疗诊断辅助

LLM可以辅助医生进行初步诊断，提供可能的诊断建议和相关信息。

```python
import openai

class MedicalDiagnosisAssistant:
    def __init__(self, api_key):
        openai.api_key = api_key

    def analyze_symptoms(self, symptoms, patient_info):
        prompt = f"""
        Based on the following patient information and symptoms, provide possible diagnoses and recommendations:

        Patient Information:
        - Age: {patient_info['age']}
        - Gender: {patient_info['gender']}- Medical History: {patient_info['medical_history']}

        Symptoms:
        {symptoms}

        Please provide:
        1. Possible diagnoses (list top 3)
        2. Recommended tests or examinations
        3. Potential treatment options
        4. Advice for the patient
        5. Any red flags or urgent concerns

        Medical Analysis:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

    def drug_interaction_check(self, medications):
        prompt = f"""
        Check for potential drug interactions among the following medications:

        {', '.join(medications)}

        Please provide:
        1. Identified potential interactions
        2. Severity of each interaction
        3. Recommendations for managing interactions
        4. Alternative medications to consider

        Drug Interaction Analysis:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=400,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

# 使用示例
medical_assistant = MedicalDiagnosisAssistant("your-openai-api-key")

patient_info = {
    "age": 45,
    "gender": "Female",
    "medical_history": "Hypertension, Type 2 Diabetes"
}

symptoms = """
- Persistent headache for the past 3 days
- Blurred vision in the right eye
- Nausea and occasional vomiting
- Sensitivity to light
"""

diagnosis = medical_assistant.analyze_symptoms(symptoms, patient_info)
print("Medical Diagnosis Assistant:")
print(diagnosis)

medications = ["Metformin", "Lisinopril", "Aspirin", "Simvastatin"]
interaction_check = medical_assistant.drug_interaction_check(medications)
print("\nDrug Interaction Check:")
print(interaction_check)
```

### 13.3.2 病历摘要生成

LLM可以用于自动生成病历摘要，帮助医疗专业人员快速了解患者情况。

```python
import openai

class MedicalRecordSummarizer:
    def __init__(self, api_key):
        openai.api_key = api_key

    def summarize_medical_record(self, medical_record):
        prompt = f"""
        Summarize the following medical record, highlighting key information:

        {medical_record}

        Please provide a summary including:
        1. Patient demographics
        2. Chief complaint
        3. Relevant medical history
        4. Key findings from physical examination
        5. Diagnostic test results
        6. Diagnosis
        7. Treatment plan
        8. Follow-up recommendations

        Medical Record Summary:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=400,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

    def generate_discharge_summary(self, admission_details, treatment_course, discharge_plan):
        prompt = f"""
        Generate a discharge summary based on the following information:

        Admission Details:
        {admission_details}

        Treatment Course:
        {treatment_course}

        Discharge Plan:
        {discharge_plan}

        Please include:
        1. Reason for admission
        2. Significant findings
        3. Procedures performed
        4. Hospital course
        5. Discharge diagnosis
        6. Discharge medications
        7. Follow-up instructions
        8. Patient education provided

        Discharge Summary:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

# 使用示例
summarizer = MedicalRecordSummarizer("your-openai-api-key")

medical_record = """
Patient: John Doe
Age: 58
Gender: Male
Chief Complaint: Chest pain and shortness of breath
Medical History: Hypertension, Hyperlipidemia
Physical Examination: BP 150/90, HR 88, RR 20, T 37.2°C
ECG: ST-segment elevation in leads V2-V4
Troponin I: Elevated (0.5 ng/mL)
Diagnosis: Acute Myocardial Infarction (STEMI)
Treatment: Emergency PCI with stent placement in LAD
Medications: Aspirin, Clopidogrel, Atorvastatin, Metoprolol
Follow-up: Cardiology clinic in 2 weeks
"""

summary = summarizer.summarize_medical_record(medical_record)
print("Medical Record Summary:")
print(summary)

admission_details = "58-year-old male admitted for acute chest pain and shortness of breath."
treatment_course = "Diagnosed with STEMI. Underwent emergency PCI with stent placement in LAD. Post-procedure course uncomplicated."
discharge_plan = "Discharge on day 5 post-PCI. Prescribed dual antiplatelet therapy, statin, and beta-blocker."

discharge_summary = summarizer.generate_discharge_summary(admission_details, treatment_course, discharge_plan)
print("\nDischarge Summary:")
print(discharge_summary)
```

### 13.3.3 医学文献智能检索

LLM可以辅助医疗专业人员进行医学文献检索，快速找到相关研究和证据。

```python
import openai
import requests
from bs4 import BeautifulSoup

class MedicalLiteratureSearch:
    def __init__(self, api_key):
        openai.api_key = api_key

    def search_pubmed(self, query, max_results=5):
        base_url = "https://pubmed.ncbi.nlm.nih.gov/"
        search_url = f"{base_url}?term={query}&size={max_results}"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        for article in soup.find_all('article', class_='full-docsum'):
            title = article.find('a', class_='docsum-title').text.strip()
            authors = article.find('span', class_='full-authors').text.strip()
            abstract_link = base_url + article.find('a', class_='docsum-title')['href']
            results.append({"title": title, "authors": authors, "link": abstract_link})
        
        return results

    def analyze_abstract(self, abstract):
        prompt = f"""
        Analyze the following medical research abstract:

        {abstract}

        Please provide:
        1. Main research question or objective
        2. Methodology used
        3. Key findings
        4. Limitations of the study
        5. Potential clinical implications

        Abstract Analysis:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=400,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

    def summarize_multiple_studies(self, abstracts):
        prompt = f"""
        Summarize and compare the following medical research abstracts:

        {abstracts}

        Please provide:
        1. Common themes or research areas
        2. Consistencies in findings across studies
        3. Any contradictions or differences in results
        4. Overall implications for clinical practice
        5. Suggestions for future research

        Multi-Study Summary:
        """

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

# 使用示例
literature_search = MedicalLiteratureSearch("your-openai-api-key")

query = "COVID-19 vaccine efficacy"
search_results = literature_search.search_pubmed(query)

print("PubMed Search Results:")
for result in search_results:
    print(f"Title: {result['title']}")
    print(f"Authors: {result['authors']}")
    print(f"Link: {result['link']}")
    print()

# 假设我们获取了第一篇文章的摘要
abstract = """
Background: The efficacy of COVID-19 vaccines in real-world settings is of utmost importance.
Methods: We conducted a retrospective cohort study of 100,000 individuals to assess vaccine efficacy.
Results: The vaccine showed 95% efficacy in preventing symptomatic COVID-19.
Conclusion: Our findings support the widespread use of the vaccine to control the pandemic.
"""

analysis = literature_search.analyze_abstract(abstract)
print("Abstract Analysis:")
print(analysis)

# 假设我们有多个研究的摘要
multiple_abstracts = """
Study 1: [Abstract of the first study]
Study 2: [Abstract of the second study]
Study 3: [Abstract of the third study]
"""

multi_study_summary = literature_search.summarize_multiple_studies(multiple_abstracts)
print("\nMulti-Study Summary:")
print(multi_study_summary)
```

这些示例展示了LLM在金融、法律和医疗健康等垂直领域的潜在应用。通过结合领域特定知识和LLM的强大能力，我们可以创建智能辅助工具，提高专业人士的工作效率和决策质量。在实际应用中，这些系统还需要进一步的定制、验证和与专业知识的整合，以确保其输出的准确性和可靠性。
