import re

def fix_bib_fields():
    bib_path = 'c:/Tugas_Akhir/ARGOS_public_anon/tugas_akhir/config/references.bib'
    with open(bib_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    def replacer(match):
        field_name = match.group(1)
        value = match.group(2)
        # remove surrounding quotes or braces
        value = value.strip()
        if (value.startswith('{') and value.endswith('}')) or (value.startswith('"') and value.endswith('"')):
            value = value[1:-1]
        
        # Remove any existing multiple braces at the edges
        value = value.strip('{}')
        
        # Re-wrap in {{{...}}}
        return f"{field_name} = {{{{{{{value}}}}}}},"
        
    # Match booktitle or journal fields
    # Example: booktitle={Proceedings of ...},
    # Example: journal = "Some Journal",
    
    new_content = re.sub(r'(?im)^(\s*(?:booktitle|journal))\s*=\s*(.*?)[,]?\s*$', replacer, content)
    
    with open(bib_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print("Fixed booktitle and journal fields in references.bib")

if __name__ == "__main__":
    fix_bib_fields()
