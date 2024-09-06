import streamlit as st
import xml.etree.ElementTree as et
import nltk
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity
from oxford import *
from wordnet import *
from compare import *
# Ensure you have the necessary nltk data
nltk.download('wordnet')

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'previous_search' not in st.session_state:
    st.session_state.previous_search = []

# Streamlit UI
st.title("Oxford Dictionary and WordNet Integration")

st.markdown("""
## 📚 User Guide

Welcome to the Oxford Dictionary and WordNet Integration tool! Here’s a quick guide to help you navigate through the application:

### **1. How to Use This Tool:**

1. **Upload Your XML File:**
   - Click the "Upload XML file" button to upload the XML file containing the data you want to analyze. This file should include the necessary information about words and their meanings.

2. **Enter a Word:**
   - Type the word you want to search for in the text input field labeled "Enter a word:".

3. **Submit:**
   - After entering the word, click the "Submit" button. The application will process your request and compare the data from the XML file with the Oxford Dictionary and WordNet.

### **2. What You’ll See:**

- **Matched Results:**
  - The application will display the results where the information from the XML file and Oxford Dictionary matches with WordNet. You’ll see details such as Synset ID, Pronunciation, Word Type, Description, and Examples from both sources.

- **Unmatched Results:**
  - If there are any entries that do not have corresponding matches, they will be displayed under the "Unmatched Results" section. This is split into two parts:
    - **Unmatched meanings from the XML file**: Details from the XML file that could not be matched with WordNet.
    - **Unmatched meanings from WordNet**: Details from WordNet that could not be matched with the XML file.

### **3. Data Sources:**

- **[Data here](https://drive.google.com/drive/folders/1qv-zackziQsDyRqkbkgn5NfhMFxJQSK2?usp=sharing)**: Contains the data used for analysis.
- **[Oxford Dictionary here](https://drive.google.com/drive/folders/10XpFyTV1IAaF_o4RcAYXCroVOE744W0r?usp=sharing)**: Contains additional data from the Oxford Dictionary.

### **4. Tips:**

- Ensure that the XML file is correctly formatted and contains the necessary information for accurate comparisons.
- Double-check the spelling of the word you enter to get accurate results.
- Review both matched and unmatched results to understand the differences between the sources.

Happy Analyzing!
""")

st.write(f"⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ")

# Upload XML file
uploaded_file = st.file_uploader("Upload XML file", type="xml")

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

# Check if a file has been uploaded
if st.session_state.uploaded_file is not None:
    tree = et.parse(st.session_state.uploaded_file)
    root = tree.getroot()

    # Input field for entering a word
    word = st.text_input("Enter a word:")

    # Button for submitting the word
    if st.button("Submit"):
        if word:
            # Retrieve information based on the entered word
            thongtin = thongtin1tu(word, root)
            dictionary_info = create_english_dictionary(word)
            similar_pairs = find_high_similarity_pairs(thongtin, dictionary_info)

            # Prepare lists for unmatched meanings
            matched_thongtin_indices = {pair['Index_List1'] for pair in similar_pairs}
            matched_dictionary_indices = {pair['Index_List2'] for pair in similar_pairs}

            unmatched_thongtin = [item for idx, item in enumerate(thongtin) if idx not in matched_thongtin_indices]
            unmatched_dictionary = [item for idx, item in enumerate(dictionary_info) if idx not in matched_dictionary_indices]

            # Store search results in session state
            st.session_state.previous_search.append({
                'word': word,
                'similar_pairs': similar_pairs,
                'unmatched_thongtin': unmatched_thongtin,
                'unmatched_dictionary': unmatched_dictionary
            })

            # Display matched results
            st.subheader("🔍 Matched Results")
            for pair in similar_pairs:
                index1 = pair['Index_List1']
                index2 = pair['Index_List2']

                item1 = thongtin[index1]
                item2 = dictionary_info[index2]

                st.write(f"**🧩 Synset_ID from WordNet:** {item2.get('Synset_ID', 'N/A')}")
                st.write(f"**📚 Pronunciation from Oxford Dictionary:** {''.join(item1.get('Pronunciation', []))}")
                st.write(f"**📝 Word_Type:** {item1.get('Word_Type', 'N/A')}")
                st.write(f"**📖 Description from Oxford Dictionary:** {item1.get('Description', 'N/A')}")
                st.write(f"**📜 Description from WordNet:** {item2.get('Description', 'N/A')}")
                st.write(f"**🌐 Meaning_Vietnamese from Oxford Dictionary:** {item1.get('Meaning_Vietnamese', 'N/A')}")

                # Display examples from Oxford Dictionary
                examples_oxford = item1.get('Examples', [])
                st.write("**🗣️ Examples from Oxford Dictionary:**")
                if examples_oxford:
                    st.write('\n'.join(f"- {example}" for example in examples_oxford))
                else:
                    st.write("None")

                # Display examples from WordNet
                st.write("**📚 Examples from WordNet:**")
                examples_wordnet = item2.get('Examples', [])
                if examples_wordnet:
                    st.write('\n'.join(f"- {example}" for example in examples_wordnet))
                else:
                    st.write("None")

                st.write(f"**⭐ Similarity Score:** {pair.get('Similarity', 0.0):.2f}")
                st.write(f"---\n")

            st.write(f"⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ")

            # Display unmatched results
            st.subheader("❌ Unmatched Results")

            # Display unmatched meanings from XML
            st.subheader("**❌ Unmatched meanings from XML:**")
            for item in unmatched_thongtin:
                st.write(f"**📖 Pronunciation:** {''.join(item.get('Pronunciation', []))}")
                st.write(f"**📝 Word_Type:** {item.get('Word_Type', 'N/A')}")
                st.write(f"**📜 Description:** {item.get('Description', 'N/A')}")
                st.write(f"**🌐 Meaning_Vietnamese:** {item.get('Meaning_Vietnamese', 'N/A')}")
                st.write("**🗣️ Examples:**")
                if item.get('Examples'):
                    st.write('\n'.join(f"- {example}" for example in item['Examples']))
                else:
                    st.write("None")
                st.write(f"---\n")

            st.write(f"⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ")

            # Display unmatched meanings from WordNet
            st.subheader("**❌ Unmatched meanings from WordNet:**")
            for item in unmatched_dictionary:
                st.write(f"**🧩 Synset_ID:** {item.get('Synset_ID', 'N/A')}")
                st.write(f"**📜 Description:** {item.get('Description', 'N/A')}")
                st.write(f"**🔑 Synonyms:** {', '.join(item.get('Synonyms', []))}")
                st.write(f"**🚫 Antonyms:** {', '.join(item.get('Antonyms', []))}")
                st.write(f"**👨‍👩‍👧‍👦 Parents:** {', '.join(item.get('Parents', []))}")
                st.write(f"**👶 Children:** {', '.join(item.get('Children', []))}")
                st.write("**🗣️ Examples:**")
                if item.get('Examples'):
                    st.write('\n'.join(f"- {example}" for example in item['Examples']))
                else:
                    st.write("None")
                st.write(f"---\n")
        else:
            st.warning("⚠️ Please enter a word.")

    # Display previous searches
    if st.session_state.previous_search:
        st.write(f"⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ⭐ ")
        st.subheader("🔍 Previous Searches")
        for search in st.session_state.previous_search:
            st.write(f"**🔤 Word Searched:** {search['word']}")
            st.write(f"**🔍 Matched Results:**")
            for pair in search['similar_pairs']:
                index1 = pair['Index_List1']
                index2 = pair['Index_List2']

                item1 = thongtin1tu(search['word'], root)[index1]
                item2 = create_english_dictionary(search['word'])[index2]

                st.write(f"**🧩 Synset_ID from WordNet:** {item2.get('Synset_ID', 'N/A')}")
                st.write(f"**📚 Pronunciation from Oxford Dictionary:** {''.join(item1.get('Pronunciation', []))}")
                st.write(f"**📝 Word_Type:** {item1.get('Word_Type', 'N/A')}")
                st.write(f"**📖 Description from Oxford Dictionary:** {item1.get('Description', 'N/A')}")
                st.write(f"**📜 Description from WordNet:** {item2.get('Description', 'N/A')}")
                st.write(f"**🌐 Meaning_Vietnamese from Oxford Dictionary:** {item1.get('Meaning_Vietnamese', 'N/A')}")

                # Display examples from Oxford Dictionary
                examples_oxford = item1.get('Examples', [])
                st.write("**🗣️ Examples from Oxford Dictionary:**")
                if examples_oxford:
                    st.write('\n'.join(f"- {example}" for example in examples_oxford))
                else:
                    st.write("None")

                # Display examples from WordNet
                st.write("**📚 Examples from WordNet:**")
                examples_wordnet = item2.get('Examples', [])
                if examples_wordnet:
                    st.write('\n'.join(f"- {example}" for example in examples_wordnet))
                else:
                    st.write("None")

                st.write(f"**⭐ Similarity Score:** {pair.get('Similarity', 0.0):.2f}")
                st.write(f"---\n")
