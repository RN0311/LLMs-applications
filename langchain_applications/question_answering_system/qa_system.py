from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return text.split("References")[0].strip() if "References" in text else text.strip()

def setup_qa_chain(openai_api_key):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        openai_api_key=openai_api_key
    )
    
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Context: {context}\nQuestion: {question}\nAnswer:"
    )
    
    return LLMChain(llm=llm, prompt=qa_prompt)

if __name__ == "__main__":
    OPENAI_API_KEY = "" #use your own openai api key here
    PDF_PATH = "/content/deepseek.pdf"
    
    pdf_text = extract_text_from_pdf(PDF_PATH)[:3000]
    qa_chain = setup_qa_chain(OPENAI_API_KEY)
    
    questions = [
        "What are the key contributions of this research paper?",
        "What challenges did DeepSeek-R1-Zero face?",
        "How does DeepSeek-R1 perform compared to OpenAI's models?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        response = qa_chain.invoke({"context": pdf_text, "question": question})
        print(f"Answer: {response['text']}")