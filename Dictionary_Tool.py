# Imports
import json
from dash import Dash, dcc, html, Input, Output, State, callback
import docx2txt
from string import ascii_letters, punctuation
import re
from fuzzywuzzy import process
import base64, io

# App Code
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow only one file to be uploaded
            multiple=False
        ),
        dcc.Store(id='intermediate-value'),
        html.I("\Press Enter and/or Tab key in Input to cancel the delay"),
        html.Br(),
        dcc.Input(id="input", type="text", placeholder="", debounce=True),
        html.Div(id='output', style={'whiteSpace': 'pre-line'})
    ]
)

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

# Global Variable Definition
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
punc = re.compile("[^" + re.escape(punctuation) + "]")

# Helper Functions
def process_word_doc(document):
    '''
    Returns a string object of the desired file.
    '''
    return docx2txt.process(document)

def split_documents(text):
    '''
    Splits the text in places where at least 4 newline characters appear.
    @param text: string
    '''
    return text.split('\n\n\n\n')

def delete_blanks(text):
    '''
    Deletes empty string entries.
    @param text: array of strings
    '''
    return [entry for entry in text if entry != '']

def remove_leading_spaces(text):
    '''
    Removes leading space characters for every entry.
    @param text: array of strings
    '''
    return [entry.lstrip() for entry in text]

def no_english_start(text):
    '''
    Should an entry start with English, gloms it onto the previous entry.
    @param text: array of strings
    '''
    updated_entries = []
    for entry in text:
        # Check if there are any non-punctuation characters
        if re.search(punc, entry):
            first_letter = entry[re.search(punc, entry).start()]
            if first_letter in ascii_letters:
                prev_entry = updated_entries[-1]
                updated_entries[-1] = prev_entry + '\n\n' + entry
            else:
                updated_entries.append(entry)
    return updated_entries

def separate_pointers(text):
    '''
    Any time we see a pointer (read as go from one dict entry to a different one), separate it into a different dictionary entry.
    @param text: array of strings
    '''
    with_pointers = []
    special_char = '☞'
    for entry in text:
        num_splits = entry.count('☞')
        while num_splits > 0:
            # find the index of the first occurence
            curr_index = entry.find('☞')
            start, end = curr_index, curr_index
    
            # find the start and end of the pointer phrase
            while start >= 0:
                if start == 0: break
                if '\n' in entry[start:start+1]:
                    start += 1
                    break
                start -= 1
            while end < len(entry):
                if end == len(entry) - 1: break
                if '\n' in entry[end:end+1]:
                    end -= 1
                    break
                end += 1
                
            # break it apart and continue the search
            before_pointer = entry[:start]
            pointer_phrase = entry[start:end+1]
            after_pointer = entry[end+1:]
    
            if before_pointer.lstrip(): with_pointers.append(before_pointer)
            with_pointers.append(pointer_phrase)
            entry = after_pointer
    
            # decrement number of times to search
            num_splits -= 1
        if entry:
            with_pointers.append(entry)
    return with_pointers

def merge_lone_examples(text):
    '''
    If we have an example in Korean that somehow got separated from its dictionary entry, add it back to the previous entry.
    @param text: array of strings
    '''
    merged_examples = []
    special_char = '☞'

    for entry in text:
        # Find where the first punctutation happens such that you know a statement of some sort occurred
        period = entry.find('.')
        question = entry.find('?')
        exclaim = entry.find('!')
    
        if period + question + exclaim == -3:
            merged_examples.append(entry)
            continue
    
        if period == -1: period = len(entry)
        if question == -1: question = len(entry)
        if exclaim == -1: exclaim = len(entry)
    
        breakpoint = min(period, question, exclaim)
        before_break = entry[:breakpoint]
        after_break = entry[breakpoint+1:]
    
        # Check whether there was any english before it
        # if yes, this isn't a stray example
        # NOTE OF MODIFICATION: also checking for numerical value before the period, since it could be a numbering and I don't want that split
        if not re.search('[a-zA-Z0-9]', before_break) and not special_char in before_break:
            prev_entry = merged_examples[-1]
            merged_examples[-1] = prev_entry + '\n\n' + entry
        else:
            merged_examples.append(entry)

    return merged_examples

def merge_lone_pointers(text):
    '''
    If we have an entry beginning with a pointer, add it back to the previous entry.
    @param text: array of strings
    '''
    special_char = '☞'
    no_lone_pointers = []
    for entry in text:
        if entry[0] == special_char:
            prev_entry = no_lone_pointers[-1]
            no_lone_pointers[-1] = prev_entry + '\n\n' + entry
        else:
            no_lone_pointers.append(entry)
    return no_lone_pointers

def get_keys(text):
    '''
    Separate the dicionary keys from its values. For entries containing no English (special pointer phrases), keys will be anything before the pointer.
    For entries containing english, this will be anything before the first linebreak or English or ending-punctuation marker.
    @param text: array of strings
    '''
    special_char = '☞'
    grammar_dictionary = {}

    # NOTE TO SELF: WE CAN HAVE BADLY BEHAVING POINTERS LIKE THIS '→'
    
    for entry in text:
        # Check for pointer-only entries
        if not re.search('[a-zA-Z0-9]', entry):
            #splits = entry.split(special_char)
            splits = entry.replace('→', '☞').split(special_char)
            key = clean_key(splits[0].rstrip())

            # Create new entry if not exist in dictionary
            if not key in grammar_dictionary:
                grammar_dictionary[key] = []
            grammar_dictionary[key].append(entry)
        # Otherwise, find the first part where the phrase breaks for a number, linebreak, or english
        else:
            key_end_index = re.search(r'[a-zA-z0-9\n\t\xa0.]', entry).start()
            key = clean_key(entry[:key_end_index].rstrip())

            # Create new entry if not exist in dictionary
            if not key in grammar_dictionary:
                grammar_dictionary[key] = []
            grammar_dictionary[key].append(entry)

    return grammar_dictionary

def clean_key(string):
    '''
    Cleans a dictionary key from unnecessary extraneous characters.

    @param string: dictionary key
    '''
    # Removing random hashtags, arrows, open parenth/brackets/quotes
    clean_key = string.replace('→', '☞').lstrip('#').strip('🡪').rstrip('(').rstrip('[').rstrip('“').rstrip('-').rstrip('‘')

    # Sometimes we have the pointer-phrases followed by English, so the original separation fails
    special_char = '☞'
    if special_char in clean_key:
        clean_key = clean_key[:clean_key.index(special_char)]
    clean_key = clean_key.lstrip().rstrip()

    return clean_key

def search_dictionary(dictionary, entry):
    '''
    Searches the dictionary and returns relevant entries.
    NOTE: at the moment, it's anything that matches the entry exactly

    @param dictionary: the dictionary (literal python dict) to search through
    @param entry: the dictionary key/phrase to search for
    '''
    if entry in dictionary:
        output = '\n\n'.join(dictionary[entry])
        return output
    else:
        return 'No such entry exists.'
    
def remove_dashes_tildes(string):
    '''
    Takes a string and returns the set of words contained excluding dashes and tildes.

    @param string: a string, generally dictionary entry or search term
    '''
    words = string.split(' ')
    entry_set = set()
    for word in words:
        parenth_split = word.split('/')
        non_signed_word = '/'.join([remove_dash.strip('-').strip('~') for remove_dash in parenth_split])
        entry_set.add(non_signed_word)
    return entry_set

def make_entry_set(dictionary):
    '''
    Given a dictionary, return another python dictionary which maps entry to the set of words contained.

    @param dictionary: the dictionary (literal python dict) to search through
    '''
    entry_mapping = {}
    for entry in dictionary.keys():
        entry_mapping[entry] = remove_dashes_tildes(entry)
    return entry_mapping

def categorize_entries(entry_mapping):
    '''
    Given a dictionary, return another one that contains which entries are supersets of a given key.
    Example: '가다' is a subset of both '가다' and '-어/아 가다'.

    @param entry_mapping: the dictionary (literal python dict) mapping dictionary entries to their set
    '''
    entry_supersets = {} 
    for key in entry_mapping.keys():
        key_set = entry_mapping[key]
        if not key in entry_supersets:
            entry_supersets[key] = [key]
        for comparison in entry_mapping.keys():
            if comparison != key:
                comparison_set = entry_mapping[comparison]
                # compare whether the key is a subset of the comparison
                if key_set.issubset(comparison_set):
                    entry_supersets[key].append(comparison)

    return entry_supersets

def nearest_matches(dictionary, entry_mapping, search, list_all = False, num_listed = 5):
    '''
    Given a search by the user, output the nearest 3-4 matches based on the dictionary entries.
    Edit from meeting: this will output the desired number of matches for the searched dictionary word.

    @param dictionary: the dictionary (literal python dict) to search through
    @param entry_mapping: the dictionary mapping a dictionary entry to its string decomposition
    @param search: the input of the user
    @param list_all: boolean value. If true, output every single entry where the search is a direct subset of the entry
    @param num_listed: if not list_all, output the number of matches desired by the user

    NOTE FOR CONCERN: going to have to test the dashes and squiggles in front of entries, such as -에 and ~(으)ㄴ/는/(으)ㄹ
    Idk if I should be worried about the other punctuation atm, but I'll strip it later for search terms? CIRCLE BACK TO THIS
    '''
    nearest_matches = []
    
    # Get the set of words for the search term
    search_set = remove_dashes_tildes(search)

    # Check to see how many supersets we've got in the dictionary
    # Most importantly, check whether this already exists in our dictionary
    # Note to self - eventually we can use the subset search feature for faster querying. Atm deleted entry_supersets, replaced with entry_mapping
    for entry in entry_mapping.keys():
        if search_set == entry_mapping[entry]:
            nearest_matches.insert(0, entry)
        elif search_set.issubset(entry_mapping[entry]): 
            nearest_matches.append(entry)

    # Beyond that, do fuzzy string matching
    fuzzy_matches = process.extract(search, list(dictionary.keys()))

    # Merge results
    index = 0
    while len(nearest_matches) < num_listed and index < len(fuzzy_matches):
        if not fuzzy_matches[index][0] in nearest_matches:
            nearest_matches.append(fuzzy_matches[index][0])
        index += 1

    if list_all:
        return nearest_matches
    return nearest_matches[:min(num_listed, len(nearest_matches))]

def print_results(dictionary, search, nearest_matches):
    '''
    For the given search term, output the dictionary definitions of the nearest matches.

    @param dictionary: the dictionary (literal python dict) to search through
    @param search: the input of the user
    @param nearest_matches: dictionary entries that match the search term most closely
    '''
    linebreak = '\n\n ---------------------------------------------------------------- \n\n'

    output = 'Search: ' + search + linebreak
    for match in nearest_matches:
        output += search_dictionary(dictionary, match)
        output += linebreak
    return output

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    file_path = io.BytesIO(decoded)
    try:
        if 'docx' in filename:
            # Assume that the user uploaded a word file
            word_file = process_word_doc(file_path)
            test_split_file = split_documents(word_file)
            
            test_blanks = remove_leading_spaces(test_split_file)
            test_blanks = delete_blanks(test_blanks)
            
            test_pointers = separate_pointers(test_blanks)
            test_pointers = remove_leading_spaces(test_pointers)
            test_pointers = delete_blanks(test_pointers)
            
            test_merge = merge_lone_pointers(test_pointers)
            test_merge = no_english_start(test_merge)
            test_merge = merge_lone_examples(test_merge)

            test_dict = get_keys(test_merge)
            test_mapping = make_entry_set(test_dict)

            datasets = {
                'dictionary': test_dict, #test_dict.to_json(orient='split', date_format='iso'),
                'mapping': test_mapping, #test_mapping.to_json(orient='split', date_format='iso'),
            }
            print('Upload Successful')
            return html.Div([json.dumps(datasets, cls=SetEncoder)])
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
        
@callback(Output('intermediate-value', 'data'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        return parse_contents(list_of_contents, list_of_names, list_of_dates)


@callback(
    Output("output", "children"),
    [Input("input", "value"),
    Input("intermediate-value", "data")],
)
def update_output(input, data):
    print(input)
    if input:
        datasets = json.loads(data['props']['children'][0])
        test_dict = datasets['dictionary']
        test_mapping = datasets['mapping'] 
        matching_outputs = nearest_matches(test_dict, test_mapping, input)
        results = print_results(test_dict, input, matching_outputs)
        return '{}'.format(results)
    else:
        return "No current search"

if __name__ == "__main__":
    app.run_server(debug=False)
