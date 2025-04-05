import PyPDF2

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from all pages of a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        PyPDF2.PdfReadError: If there's an error reading the PDF
    """
    try:
        # Open the PDF file in binary read mode
        with open(pdf_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Initialize an empty string to store the text
            text = ""
            
            # Iterate through all pages and extract text
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
                
            return text.strip()
            
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {pdf_path} was not found")
    except PyPDF2.PdfReadError as e:
        raise PyPDF2.PdfReadError(f"Error reading PDF file: {str(e)}")

text=extract_text_from_pdf(r"E:\RFP-Analyzer\NEW_RESUME (7).pdf")
print(text)
