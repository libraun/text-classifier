import re

# Match anything after a "Reference" (or "External Links") header
CHAR_SUB_EXPR = re.compile("[^\x00-\x7F]+")

# Match anything after a "Reference" (or "External Links") header
ENDTEXT_EXPR = re.compile("==+ +?[References|External Links][\\S\\s]*")

# Match common JavaScript and header plaintext tags
WIKITAG_EXPR = re.compile("==+[^=]*?==+|\\n|\\r|{[^}]*?}")

# Match more than one space
EXTRASPACE_EXPR = re.compile("  +")

def preprocess_text(text: str, repl: str=" ") -> str:

    # Convert text to lowercase
    text = text.lower()

    # Remove wikipedia-specific tags/unnecessary text
    text = re.sub(ENDTEXT_EXPR, repl, text)
    text = re.sub(WIKITAG_EXPR, repl, text)

    # Remove all extra spaces and non-unicode characters
    text = re.sub(EXTRASPACE_EXPR, repl, text)

    text = text.replace(r'\\', '')
    #text = re.sub(BADCHAR_EXPR, repl, text)

    return text





        
     
      
      


