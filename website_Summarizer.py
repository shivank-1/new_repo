import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables

# A class to represent a Webpage
class Website:
    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""

    def get_contents(self):
        return {"title": self.title, "text": self.text}

# Function to retrieve webpage details
def get_all_details(url):
    page = Website(url)
    return page.get_contents()

import os
from dotenv import load_dotenv
load_dotenv()
groqkey = os.getenv('GROQKEY')
# Load LLM
api_key = groqkey

llm = ChatGroq(groq_api_key=api_key, model="llama3-8b-8192", temperature=0.5)

# Streamlit UI
st.title("CrystalClear Web Insights")
st.subheader("Effortlessly Generate Comprehensive Summarized Notes from Website Articles. Paste the Website Link Below to Get Started")
website_link = st.text_input("Paste the Website URL")

def generate_response(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    
    # Wrap transcript in a Document object
    document = Document(page_content=text)
    
    # Split the document
    splits = text_splitter.split_documents([document])
    
    # Debugging: Uncomment to see splits
    #st.write(splits)

    chunks_prompt = """
    Please summarize the below text:
    Text: "{text}"
    Summary:
    """

    map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)

    final_prompt = '''
    Provide comprehensive and detailed notes based on the given documents, focusing on key concepts, explanations, and examples. The notes should help college students not only understand the topics thoroughly but also prepare effectively for their exams. Ensure the content is organized, with clear headings, subheadings, bullet points, and concise explanations where necessary.
    Documents: {text}
    '''

    final_prompt_template = PromptTemplate(input_variables=['text'], template=final_prompt)

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=final_prompt_template,
        verbose=True
    )

    output = summary_chain.run(splits)
    return output


if st.button("Summarize the Content"):
    if website_link:
        with st.spinner("Processing..."):
            try:
                webpage_details = get_all_details(website_link)
                webpage_details=str(webpage_details)
                summary = generate_response(webpage_details)
                st.markdown("## Detailed Notes:")
                st.write(summary)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please provide a valid website link.")