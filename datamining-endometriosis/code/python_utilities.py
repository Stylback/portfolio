#-------------------
# Prerequisite modules

import os
import re
import pickle
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi
import warnings
from pandas.plotting import table
from scipy.stats import chi2, chi2_contingency
from matplotlib.ticker import StrMethodFormatter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from deep_translator import GoogleTranslator

# Surpress warning messages of unoptimized dataframe operations
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Get and set directory
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
data_path = parent_dir + '/data/'
media_path = parent_dir + '/media/'

#-------------------

# Help function to sort list of strings as if they were int
def asint(s):
    try: return int(s), ''
    except ValueError: return sys.maxsize, s

def extract_questionnaire_structure(input_file: str, output_file: str):
    
    """
    This function will extract question and answer details from "FemaleQuestionnaire.kt", aggregate the details in a dictionary and dump them to a file.
    The resulting dictionary will be on the form:
        {question_id:{'questionType', 'val', 'localizedQuestionTextMap', 'answer_alternatives': {}}}

    Arguments:
        input_file {str} -- Filename of input file, e.g., "file.kt".
        output_file {str} -- Filename of output file, e.g., "file.db".
    
    Returns:
        None
    """
    
    with open(data_path + input_file, mode='r') as in_file, \
    open(data_path + output_file, mode='wb') as out_file:
  
        in_file_read = in_file.read()

        answer_index = in_file_read.index("object Answers {")
        question_index = in_file_read.index("object Questions {")

        raw_textmaps = in_file_read[:answer_index].replace('\n','')
        raw_textmaps = raw_textmaps.split("val ")[2:]

        raw_answers = in_file_read[answer_index:question_index].replace('\n','')
        raw_answers = raw_answers.split("object ")[2:]

        raw_questions = in_file_read[question_index:-1].replace('\n','')
        raw_questions = raw_questions.split("val ")[1:]

        textmap_dict = {}
        textmap_pattern = r'(.*?) = mapOf\("en" to "(.*?)", "hu" to "(.+?)"\)'

        for textmap in raw_textmaps:
            matches = re.findall(textmap_pattern, textmap)
            if len(matches) != 0:
                localizedAnswerTextMap = matches[0][0]
                eng = matches[0][1]
                hu = matches[0][2]
                textmap_dict[localizedAnswerTextMap] = ({'eng': eng, 'hu': hu})

        answer_dict = {}
  
        object_pattern = r'(.*?) { \s+val .+? = AnswerInfo.PredefinedAnswerInfo'
        answer_pattern = r'val (.+?) = AnswerInfo.PredefinedAnswerInfo\(\s+id = (\d{1,3}),\s+ localizedAnswerTextMap = (.+?), \s+ numericalValue = (\d{1,3})'

        for answer in raw_answers:
            matches = re.findall(answer_pattern, answer)
            object_match = re.findall(object_pattern, answer)
            for match in matches:
                val = str(object_match[0]) + '.' + str(match[0])
                id = match[1]
                localizedAnswerTextMap = textmap_dict[match[2]]['eng']
                numericalValue = match[3]
                answer_dict[val] = ({'id': id, 'localizedAnswerTextMap':localizedAnswerTextMap, 'numericalValue':numericalValue})
        
        questions_dict = {}
        questions_pattern = r'(.+?) = Question\.(.+?)\(\s+ id = (\d{1,3}),\s+ localizedQuestionTextMap = (.+?),.+?'

        for question in raw_questions:
            matches = re.findall(questions_pattern, question)
            if len(matches) != 0:
                val = matches[0][0]
                questionType = matches[0][1]
                id = matches[0][2]
                localizedQuestionTextMap = textmap_dict[matches[0][3]]['eng']
                answer_alternatives = {}

                match questionType:
                    case 'MultiChoiceQuestion' | 'SingleChoiceQuestion':
                        answer_pattern = r'.+? Answers\.(.+?\..+?)\.id'
                        matches = re.findall(answer_pattern, question)
                        for match in matches:
                            answer_id = answer_dict[match]['id']
                            label = answer_dict[match]['localizedAnswerTextMap']
                            answer_alternatives[answer_id] = ({'label':label})
                        
                    case 'ValueRangeQuestion':
                        answer_pattern = r'.+? rangeMin = (\d{1,3}),\s+rangeMax = (\d{1,3}),'
                        matches = re.findall(answer_pattern, question)
                        answer_alternatives.update({'rangeMin':matches[0][0]})
                        answer_alternatives.update({'rangeMax':matches[0][1]})

                    case 'NumericQuestion':
                        answer_pattern = r'.+? minLength = (\d{1,3}),\s+maxLength = (\d{1,3}),'
                        matches = re.findall(answer_pattern, question)
                        answer_alternatives.update({'minLength':matches[0][0]})
                        answer_alternatives.update({'maxLength':matches[0][1]})
                            
                    case 'FreeFormQuestion'| 'MultiFreeFormQuestion':
                        answer_pattern = r'.+? maxEntries = (\d{1,3}).+?\)'
                        matches = re.findall(answer_pattern, question)
                        if len(matches) != 0:
                            answer_alternatives.update({'maxEntries':matches[0]})
                
                answer_alternatives = dict([(alt, answer_alternatives[alt]) for alt in sorted(answer_alternatives, key=asint)])
                
                questions_dict[id] = ({'questionType': questionType, 'val': val, 'localizedQuestionTextMap':localizedQuestionTextMap, 'answer_alternatives':answer_alternatives})
                
        
        pickle.dump(questions_dict, out_file)
        
        print("\nDone.")

    return

#-------------------

def process_freeform_answer(regex_match):
    
    """
    This function takes a Match object and split it into its corresponding subgroups.
    The relevant subgroup containing the freeform answer will be sanitized by removing special characters.

    Arguments:
        regex_match {object} -- A RegEx Match object, the freeform answer matched by RegEx to be processed.
    
    Returns:
        processed_answer {str} -- A string containing prepended subgroup, processed answer and appended subgroup.
    """

    # Isolate subgroups as strings
    prepend = regex_match.group(1)
    raw_answer = regex_match.group(2)
    append = regex_match.group(3)

    # Sanitize answer string
    sanitized_answer = re.sub('[^A-Za-z0-9-ÁáÉéÓóÖöŐőÚúÜüŰű\\s,]+', '', raw_answer)
    processed_answer = prepend + r'"' + sanitized_answer + r'"' + append
    
    return processed_answer

#-------------------

def process_multi_freeform_answer(regex_match):
    
    """
    This function takes a Match object and split it into its corresponding subgroups.
    The relevant subgroup containing the freeform answer will be sanitized by removing special characters.

    Arguments:
        regex_match {object} -- A RegEx Match object, the freeform answer matched by RegEx to be processed.
    
    Returns:
        processed_answers {str} -- A string containing prepended subgroup, processed answers and appended subgroup.
    """
    
    # Isolate subgroups as strings
    prepend = regex_match.group(1)
    raw_answers = regex_match.group(2)
    append = regex_match.group(3)

    # Work-around for a particular instance of escape character
    preprocessed_answers = raw_answers.replace('},"','","')

    answers_json = json.loads(preprocessed_answers)
    answers_dict = {}
    answer_keys = answers_json.keys()

    # For each key:value pair, sanitize value
    for key in answer_keys:
        value = answers_json[key]
        sanitized_subanswer = re.sub('[^A-Za-z0-9-ÁáÉéÓóÖöŐőÚúÜüŰű\\s]+', '', value)
        answers_dict.update({key: sanitized_subanswer})

    sanitized_answers = json.dumps(answers_dict)
    processed_answers = prepend + sanitized_answers + append

    return processed_answers

#-------------------

def build_dictionary(input_file: str, output_file: str):
    
    """
    This function will convert the "questionnaire_result" database into a Python dictionary on the form:
        {id:{question_id:{type, value, timestamp}, date, language, completion_count}}

    Arguments:
        input_file {str} -- Filename of input file, e.g., "answers.db".
        output_file {str} --  Filename of output file, e.g., "answers_dict.db".

    Returns:
        None
    """

    # Open input and output files
    with open(data_path + input_file, mode='r') as in_file, \
    open(data_path + output_file, mode='wb') as out_file:

        # Split lines to simplify iteration
        in_file_split = in_file.read().splitlines(True)
        
        # Initialize dictionary
        filled_questionnaire = {}

        for line in in_file_split[1:-1]:

            # Split id from the rest of the entry
            entry = line.split(",",1)
            id = entry[0]
            entry = entry[1].replace("\'\"","\"")

            # Isolate information from end of questionnaire
            trailing_info = re.findall(r'}}",(.*)', entry)
            trailing_matches = re.findall(r'.*?(\d{4}-\d{2}-\d{2}),(\w{2}),".*?",\d,(.*?),', trailing_info[0])
            date = str(trailing_matches[0][0])
            language = str(trailing_matches[0][1])
            connection_id = str(trailing_matches[0][2])
            
            # Number of operations to convert answers_json to valid json format
            answers_json = re.sub(r'}}",(.*)',r'}}', entry)
            answers_json = answers_json.replace(":\"{",":{")
            answers_json = answers_json.replace("}\",","},")
            answers_json = answers_json.replace("\\'\"","")
            answers_json = answers_json.replace("{\\'\"","")
            
            # Sanitize freeform answers
            answers_json = re.sub(r'(:{\"type\":\"freeform\",\"answer\":)(.*?[{}]?)(,\"timestamp\":\d{13}})', process_freeform_answer, answers_json)
            answers_json = re.sub(r'(:{\"type\":\"multi_freeform\",\"answers\":)(.*?[{}]?)(,\"timestamp\":\d{13}})', process_multi_freeform_answer, answers_json)
            
            # Remove "type" and "answer" from keys
            answers_json = re.sub(r'(:{\"type\":\"freeform\",\"answer\":)(.*?[{}]?)(,\"timestamp\":\d{13}})', r':{"freeform":\2\3', answers_json)
            answers_json = re.sub(r'(:{\"type\":\"multi_freeform\",\"answers\":)(.*?[{}]?)(,\"timestamp\":\d{13}})', r':{"multi_freeform":\2\3', answers_json)
            answers_json = re.sub(r'(:{\"type\":\"single\",\"predefinedAnswerInfoId\":)(.*?[{}]?)(,\"timestamp\":\d{13}})', r':{"single":\2\3', answers_json)
            answers_json = re.sub(r'(:{\"type\":\"multi\",\"predefinedAnswerInfoIds\":)(.*?[{}]?)(,\"timestamp\":\d{13}})', r':{"multi":\2\3', answers_json)
            answers_json = re.sub(r'(:{\"type\":\"value\",\"answer\":)(.*?[{}]?)(,\"timestamp\":\d{13}})', r':{"value":\2\3', answers_json)
            answers_json = re.sub(r'(:{\"type\":\"none\")(,\"timestamp\":\d{13}})', r':{"none":" "\2', answers_json)

            # Strip line breakes
            answers_json = answers_json.rstrip(",\n")
            answers_json = answers_json.lstrip("\"")
            
            # Load answers_json as a dictionary
            answers_json = json.loads(answers_json)
            
            # Add key-value pairs to dictionary
            answers_json.update({'date': date})
            answers_json.update({'language': language})
            answers_json.update({'connection_id': connection_id})

            # Nestle dictionary with id as top-level key
            filled_questionnaire[id] = answers_json
            
        # Save dictionary to file
        pickle.dump(filled_questionnaire, out_file)

        print("\nDone.")

    return

#-------------------

def build_database(input_filename: str, output_filename: str):
    
    """
    This function takes a list of nested dictionaries containing questionnaire answers and converts it to a Pandas dataframe.

    Arguments:
        input_filename {str} -- Filename of input file, e.g., "questionnaire_result.dict".
        output_filename {str} --  Filename for output file, e.g., "questionnaire_result.df".

    Returns:
        None
    """

    with open(data_path + input_filename, mode='rb') as in_file, \
        open(data_path + output_filename, mode='wb') as out_file:

        questionnaire_dict = pickle.load(in_file)
        total_questionnaires = str(len(questionnaire_dict)) # For printout

        questionnaire_ids = questionnaire_dict.keys()

        shifted_questionnaires = [] # for debugging purposes
        shifted_ids = [] # for debugging purposes
        not_shifted_questionnaires = []
        not_shifted_ids = []

        for id in questionnaire_ids:
            questionnaire = questionnaire_dict[id]
            if "{'1': {'single':" in str(questionnaire):
                shifted_questionnaires.append(questionnaire)
                shifted_ids.append(id)
            else:
                not_shifted_questionnaires.append(questionnaire)
                not_shifted_ids.append(id)
        
        amount_shifted_questionnaires = str(len(shifted_questionnaires)) # For printout
        amount_not_shifted_questionnaires = str(len(not_shifted_questionnaires)) # For printout

        df = pd.json_normalize(not_shifted_questionnaires).set_index(pd.Series(not_shifted_ids))
        df = df.sort_values(by = 'connection_id') # places questionnaires without a connection_id at the top
        df = df.drop_duplicates(subset = ['7.timestamp'], keep='last') # should retain the duplicate with a connection_id
        amount_duplicate_questionnaires = str(int(amount_not_shifted_questionnaires) - len(df)) # For printout
        unique_questionnaires = str(len(df)) # For printout

        count_freq = df['connection_id'].value_counts()
        id_grouped_df = df.groupby(by='connection_id', dropna=False).agg(list)
        id_grouped_df = pd.merge(id_grouped_df, count_freq, left_index=True, right_index=True).dropna()
        respondence_statistics = id_grouped_df['count'].value_counts(ascending=True, dropna=False).to_frame(name='')

        # Bin ID's into different diagnosis subsets
        for connection_id, answers in id_grouped_df['5.multi'].items():
            for answer in answers:
                if str(answer) != 'nan':
                    if 17 in answer:
                        id_grouped_df.at[connection_id, 'diagnosis'] = 'endometriosis'
                    elif answer and all(elem == 16 for elem in answer):
                        id_grouped_df.at[connection_id, 'diagnosis'] = 'no_diagnosis'
                    else:
                        id_grouped_df.at[connection_id, 'diagnosis'] = 'other_diagnosis'
                else:
                    id_grouped_df.at[connection_id, 'diagnosis'] = 'no_answer'
        
        diagnosis_df = pd.DataFrame(columns=['connection_id', 'diagnosis'])
        diagnosis_df['diagnosis'] = id_grouped_df['diagnosis']
        diagnosis_df['connection_id'] = id_grouped_df.index
        
        for diag_connection_id, diagnosis in diagnosis_df['diagnosis'].items():
            for questionnaire_id, df_connection_id in df['connection_id'].items():
                if str(diag_connection_id) == str(df_connection_id):
                    df.at[questionnaire_id, 'diagnosis'] = diagnosis
        
        # Check for questionnaires without connection_id
        for row, answer in df[df['connection_id'] == ""]['5.multi'].items():
            if str(answer) != 'nan':
                if 17 in answer:
                    df.at[row, 'diagnosis'] = 'endometriosis'
                elif answer and all(elem == 16 for elem in answer):
                    df.at[row, 'diagnosis'] = 'no_diagnosis'
                else:
                    df.at[row, 'diagnosis'] = 'other_diagnosis'
            else:
                df.at[row, 'diagnosis'] = 'no_answer'

        # Check for questionnaires without answers
        for row, diagnosis in df['diagnosis'].items():
            if diagnosis == "no_answer":
                if df.at[row, '180.single'] == 62:
                    df.at[row, 'diagnosis'] = "other_diagnosis"
                elif df.at[row, '180.single'] == 63:
                    df.at[row, 'diagnosis'] = "endometriosis"
                else:
                    df.at[row, 'diagnosis'] = "no_answer"
                
        remaining_questionnaires = str(len(df))
        diagnosis_count = df['diagnosis'].value_counts(dropna=False).to_frame(name='')
        df = df[df.columns.drop(list(df.filter(regex='timestamp')))]
        df = df.reindex(sorted(df.columns), axis=1)

        pickle.dump(df, out_file)
        
        print('\nDATABASE STATISTICS')
        print(total_questionnaires + ' questionnaires in the questionnaire_result file.')
        print(amount_shifted_questionnaires + ' had missmatched question, answer IDs and were removed.')
        print('Out of the remaining ' + amount_not_shifted_questionnaires + ' questionnaires, ' + amount_duplicate_questionnaires + ' duplicates were found and removed. ' + unique_questionnaires + ' questionnaires remaining.')
        print(remaining_questionnaires + ' usable questionnaires out of the original ' + total_questionnaires + '. Diagnosis subsets as follow:')
        print(diagnosis_count)
        print('\nSome answer frequency statistics:')
        print(respondence_statistics)

        print('\nDone.')

    return

#-------------------

def pre_process_database(data_filename: str, structure_filename: str, master_filename: str, food_filename: str):
    
    """
    This function prepares data for analysis in pipeline_part_2.
    It splits a Pandas dataframe into two csv-files, one containing questions 55 & 56 and another containing the rest of the questions.
    This split it necessary due to the way that multiple-answer questions are reported in the database.
    There are also some additional pre-processing done such as value remapping.

    Arguments:
        data_filename {str} -- Filename of the data input file, e.g., "questionnaire_result.df".
        structure_filename {str} --  Filename for structure input file, e.g., "questionnaire_structure.dict".
        master_filename {str} --  Filename for "master" output file, e.g., "questionnaire.csv".
        food_filename {str} --  Filename for the food output file, e.g., "food.csv".

    Returns:
        None
    """

    with open(data_path + data_filename, mode='rb') as data_file, \
        open(data_path + structure_filename, mode='rb') as structure_file, \
        open(data_path + master_filename, mode='wb') as master_file, \
        open(data_path + food_filename, mode='wb') as food_file:

        df = pickle.load(data_file)
        structure_dict = pickle.load(structure_file)

        df = df[df.columns.drop(list(df.filter(regex='timestamp')))]
        df = df[df.columns.drop(list(df.filter(regex='freeform')))]
        df = df.drop(columns=['0.none', '120.none', '180.single', '5.multi', 'date', 'language', 'connection_id']) # Does not contribute to analysis
        df = df.drop(columns=['18.single', '19.single', '20.single', '21.single', '22.single']) # Endometriosis specific questions

        food_df = pd.DataFrame(columns=['diagnosis', '55.multi', '56.multi'])
        food_df[['diagnosis', '55.multi', '56.multi']] = df[['diagnosis', '55.multi', '56.multi']]
        food_df = food_df.explode('55.multi')
        food_df = food_df.explode('56.multi')

        df = df.drop(columns=['55.multi', '56.multi'])

        for col in list(df)[:-1]:
            question = col.split('.')[0]
            question_type = col.split('.')[1]
            if question_type == "single":
                for answer in df[col]:
                    if str(answer) != "nan":
                        label = structure_dict[question]['answer_alternatives'][str(int(answer))]['label']
                        df[col] = df[col].replace(answer, label)
            elif question_type == "value":
                for answer in df[col]:
                    if str(answer) != "nan":
                        df[col] = df[col].replace(answer, str(int(answer)))
        
        df.to_csv(master_file, index=False)
        food_df.to_csv(food_file, index=False)
        
        print('\nDone.')
    
    return

#-------------------

def histogram(no_diag_df, endo_df, other_df, data_labels: list, val_col: str, title: str, xlabels: list, bins = None,):
    
    """
    This function takes three datasets and generates a histogram.

    Arguments: 
        no_diag_df {DataFrame} -- DataFrame containing the "No diagnosis" subset
        endo_df {DataFrame} -- DataFrame containing the "Endometriosis" subset
        other_df {DataFrame} -- DataFrame containing the "Other diagnosis" subset 
        data_labels {list[str]} -- List of data subset labels
        val_col {str} -- Column-name containing values
        title {str} -- Figure title
        xlabels {List[str]} -- List of strings containing labels for the x-axis
        bins {List[int]} -- Can be used to specify bins, disabled by default

    Return
        none
    """

    a_values = []
    b_values = []
    c_values = []

    question_id = val_col.split('.')[0]
    questionType = val_col.split('.')[1]
    title = title.replace('*','') + ' (Question ' + question_id + ')'

    if questionType == 'single' or questionType == 'value':
        for entry in no_diag_df[val_col].dropna():
            if str(entry) != 'nan':
                a_values.append(entry)
        
        for entry in endo_df[val_col].dropna():
            if str(entry) != 'nan':
                b_values.append(entry)
        
        for entry in other_df[val_col].dropna():
            if str(entry) != 'nan':
                c_values.append(entry)
    
    # Multi-choice questions have additional list depth
    elif questionType == 'multi':
        for entry in no_diag_df[val_col].dropna():
            for value in entry:
                if str(value) != 'nan':
                        a_values.append(value)
        
        for entry in endo_df[val_col].dropna():
            for value in entry:
                if str(value) != 'nan':
                        b_values.append(value)
        
        for entry in other_df[val_col].dropna():
            for value in entry:
                if str(value) != 'nan':
                        c_values.append(value)

    if bins is None:
        a_bins = sorted(list(set(a_values)))
        b_bins = sorted(list(set(b_values)))
        c_bins = sorted(list(set(c_values)))
    else:
        a_bins = bins
        b_bins = bins
        c_bins = bins
    
    # Have to add an additional element due to inclusion properties
    a_bins.append(max(a_bins, default=0)+1)
    b_bins.append(max(b_bins, default=0)+1)
    c_bins.append(max(c_bins)+1)

    # Get resulting height and bins
    a_heights, a_bins = np.histogram(a_values, bins=a_bins)
    b_heights, b_bins = np.histogram(b_values, bins=b_bins)
    c_heights, c_bins = np.histogram(c_values, bins=c_bins)

    a_counts = a_heights
    b_counts = b_heights
    c_counts = c_heights

    label_a = data_labels[0] + ', ' + str(len(a_values)) + ' answers'
    label_b = data_labels[1] + ', ' + str(len(b_values)) + ' answers'
    label_c = data_labels[2] + ', ' + str(len(c_values)) + ' answers'

    # Divide bin area into three, enables us to place bars on the edge of bins
    width = (a_bins[1] - a_bins[0])/3

    # Normalize height to get relative distribution
    a_heights = a_heights / len(a_values)
    b_heights = b_heights / len(b_values)
    c_heights = c_heights / len(c_values)

    t_bins = sorted(list(set([*a_bins, *b_bins, *c_bins])))

    # Figure parameters  
    plt.figure(figsize=(9, 7))
    plt.title(title, wrap=True)
    plt.xlabel('Answer alternatives')
    plt.ylabel('Distribution')
    fig_a = plt.bar(range(len(a_bins[:-1])), a_heights, -width, align='edge', color='skyblue',alpha = 0.7, label=label_a)
    fig_b = plt.bar(range(len(b_bins[:-1])), b_heights, width, align='edge', color='coral', alpha = 0.7, label=label_b)
    fig_c = plt.bar(range(len(c_bins[:-1])), c_heights, width, align='center', color='khaki', alpha = 0.7, label=label_c)
    plt.xticks(ticks = range(len(t_bins[:-1])), labels=xlabels, rotation = 45, ha="right")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.0%}'))
    plt.bar_label(fig_a, labels = a_counts, fmt='{:.1%}', fontsize=6)
    plt.bar_label(fig_b, labels = b_counts, fmt='{:.1%}', fontsize=6)
    plt.bar_label(fig_c, labels = c_counts, fmt='{:.1%}', fontsize=6)
    plt.legend(fontsize='small')

    # Save figure to file
    plt.savefig(media_path + question_id, bbox_inches='tight')

    return

#-------------------

def generate_figure(structure_filename: str, no_diag_df, endo_df, other_df, subset_labels: list, question_id: str):
    
    """
    This function takes three datasets and generates a histogram or wordcloud depending on the question ID.

    Arguments: 
        structure_filename {str} -- Filename of dictionary containing questionnaire structure details
        no_diag_df {DataFrame} -- DataFrame containing the "No diagnosis" subset
        endo_df {DataFrame} -- DataFrame containing the "Endometriosis" subset
        other_df {DataFrame} -- DataFrame containing the "Other diagnosis" subset 
        subset_labels {list[str]} -- List of data subset labels
        question_id {str} -- Question ID to generate figure for

    Return
        none
    """

    with open(data_path + structure_filename, mode='rb') as structure_file:
        
        structure_dict = pickle.load(structure_file)
        question_type = structure_dict[question_id]['questionType']
        title = structure_dict[question_id]['localizedQuestionTextMap']

        match question_type:
            case 'SingleChoiceQuestion':
                answer_alternatives = structure_dict[question_id]['answer_alternatives'].keys()
                labels = []
                for alternative in answer_alternatives:
                    labels.append(structure_dict[question_id]['answer_alternatives'][alternative]['label'])
                val_col = question_id + '.single'
                histogram(no_diag_df, endo_df, other_df, val_col=val_col, title=title, data_labels=subset_labels, xlabels=labels)

            case 'MultiChoiceQuestion':
                answer_alternatives = structure_dict[question_id]['answer_alternatives'].keys()
                labels = []
                for alternative in answer_alternatives:
                    labels.append(structure_dict[question_id]['answer_alternatives'][alternative]['label'])
                val_col = question_id + '.multi'
                histogram(no_diag_df, endo_df, other_df, val_col=val_col, title=title, data_labels=subset_labels, xlabels=labels)

            case 'ValueRangeQuestion':
                rangeMin = int(structure_dict[question_id]['answer_alternatives']['rangeMin'])
                rangeMax = int(structure_dict[question_id]['answer_alternatives']['rangeMax'])
                val_col = question_id + '.value'
                if question_id == '17':
                    labels = list(np.arange(0,101,5))
                    bins = list(np.arange(0,94,5))
                    histogram(no_diag_df, endo_df, other_df, val_col=val_col, title=title, data_labels=subset_labels, xlabels=labels, bins=bins)
                else:
                    labels = list(np.arange(rangeMin, rangeMax+1))
                    histogram(no_diag_df, endo_df, other_df, val_col=val_col, title=title, data_labels=subset_labels, xlabels=labels)

            case 'NumericQuestion':
                # Only numeric question is "1" which has a span of 0-99. This is not reflected in the Kotlin file.
                labels = list(np.arange(0,101,5))
                bins = list(np.arange(0,94,5))
                val_col = question_id + '.value'
                histogram(no_diag_df, endo_df, other_df, val_col=val_col, title=title, data_labels=subset_labels, xlabels=labels, bins=bins)
            
            case 'FreeFormQuestion':
                val_col = question_id + '.freeform'
                freeform_wordcloud(no_diag_df, endo_df, other_df, val_col=val_col, title=title)

            case 'MultiFreeFormQuestion':
                val_col = question_id + '.multi_freeform'
                freeform_wordcloud(no_diag_df, endo_df, other_df, val_col=val_col, title=title)
    return

#-------------------

# Avoids creating a new instance for each translation.
translator = GoogleTranslator(source='hu', target='en')

def translate(text: list):
    if text:
        return translator.translate_batch(text)
    else:
        return text

#-------------------

def freeform_wordcloud(no_diag_df, endo_df, other_df, val_col: str, title: str):
    
    """
    This function takes a question with freeform answers and turn it into wordclouds.

    Arguments: 
        endo_file {str} -- File containing a pandas dataframe with survey information from people with endometriosis
        healthy_file {str} -- File containing a pandas dataframe with survey information from people no diagnosis
        other_file {str} -- File containing a pandas dataframe with survey information from people with other diagosis 
        val_col {str} -- The question to which answers is to be turn into a wordcloud 
        title {str} -- Figure title

    Return
        none
    """

    question_id = val_col.split('.')[0]
    question_type = val_col.split('.')[1]

    no_diag_answers = ''
    endo_answers = ''
    other_answers = ''

    if question_type == "freeform":
        
        for word in no_diag_df.get(val_col).dropna():
            no_diag_answers += str(word) + ' '
        
        for word in endo_df.get(val_col).dropna():       
            endo_answers += str(word) + ' '
        
        for word in other_df.get(val_col).dropna():       
            other_answers += str(word) + ' '

    elif question_type == "multi_freeform":
        
        subcol = 0
        while subcol <10 :
            for word in no_diag_df.get(val_col+'.'+str(subcol)).dropna():
                no_diag_answers += str(word) + ' '
            
            for word in endo_df.get(val_col+'.'+str(subcol)).dropna():       
                endo_answers += str(word) + ' '
            
            for word in other_df.get(val_col+'.'+str(subcol)).dropna():
                other_answers += str(word) + ' '
            subcol += 1

    print(str(len(no_diag_answers)) + ' words in the "no diagnosis" subset')
    print(str(len(endo_answers)) + ' words in the "endometriosis" subset')
    print(str(len(other_answers)) + ' words in the "other diagnosis" subset')

    no_diag_wordcloud = WordCloud(max_words = 45, stopwords = None,background_color='white', collocations=True).generate(no_diag_answers)
    endo_wordcloud = WordCloud(max_words = 45, stopwords = None, background_color='white', collocations=True).generate(endo_answers)
    other_wordcloud = WordCloud(max_words = 45, stopwords = None, background_color='white', collocations=True).generate(other_answers)

    no_diag_topwords = list(no_diag_wordcloud.words_.keys())
    endo_topwords = list(endo_wordcloud.words_.keys())
    other_topwords= list(other_wordcloud.words_.keys())

    print('The top words in the "No diagnosis" subset are: ' + str(no_diag_topwords))
    print('The top words in the "endometriosis" subset are: ' + str(endo_topwords))
    print('The top words in the "other" subset are: ' + str(other_topwords))

    no_diag_translated_topwords = ''
    endo_translated_topwords = ''
    other_translated_topwords = ''

    no_diag_translated_topwords = ' '.join(translate(no_diag_topwords))
    endo_translated_topwords = ' '.join(translate(endo_topwords))
    other_translated_topwords = ' '.join(translate(other_topwords))

    stopword = ['lot', 'none','many','sometimes','little']+ list(STOPWORDS)

    no_diag_translated_wordcloud = WordCloud(max_words = 25,stopwords =stopword, background_color='white', include_numbers=True).generate(no_diag_translated_topwords)
    endo_translated_wordcloud = WordCloud(max_words = 25,stopwords =stopword, background_color='white', include_numbers=True).generate(endo_translated_topwords)
    other_translated_wordcloud = WordCloud(max_words = 25,stopwords =stopword, background_color='white', include_numbers=True).generate(other_translated_topwords)
    
    fig = plt.figure(figsize=(5, 10))

    fig.add_subplot(3, 1, 1)
    plt.imshow(no_diag_translated_wordcloud)
    plt.axis('off')
    plt.title('No diagnosis respondents', fontsize=14)

    fig.add_subplot(3, 1, 2)    
    plt.imshow(endo_translated_wordcloud)
    plt.axis('off')
    plt.title('Endometriosis respondents', fontsize=14)

    fig.add_subplot(3, 1, 3)  
    plt.imshow(other_translated_wordcloud)
    plt.axis('off')
    plt.title('Other diagnosis respondents', fontsize=14)

    title = title.replace('*','') + ' (Question ' + question_id + ')'
    plt.suptitle(title, fontsize=16, wrap=True)

    fig.savefig(media_path + question_id, bbox_inches='tight')

#-------------------

def chi_test(questionnaire_result_filename: str, questionnaire_structure_filename: str, questions: list):
    
    """
    This function takes a question with freeform answers and turn it into wordclouds.

    Arguments: 
        questionnaire_result_filename {str} -- Filename of dataframe containing questionnaire data
        questionnaire_structure_filename {str} -- Filename of dictionary containing questionnaire structure details
        questions {list[str]} -- List of questions to test

    Return
        chi_df {DataFrame} -- A Pandas DataFrame object containing test details and results
    """
    
    with open(data_path + questionnaire_result_filename, mode='rb') as data_file, \
        open(data_path + questionnaire_structure_filename, mode='rb') as structure_file:
        
        df = pickle.load(data_file)
        structure_dict = pickle.load(structure_file)

        endo_df = df[df['diagnosis'] == "endometriosis"]
        no_diag_df = df[df['diagnosis'] == "no_diagnosis"]

        chi_list = []

        for question_id in questions:
            question_type = structure_dict[question_id]['questionType']
            val_col = ""
            match question_type:
                    case 'SingleChoiceQuestion':
                        val_col = question_id + '.single'
                    case 'MultiChoiceQuestion':
                        val_col = question_id + '.multi'
                    case 'ValueRangeQuestion':
                        val_col = question_id + '.value'
                    case 'NumericQuestion':
                        val_col = question_id + '.value'
                    case 'FreeFormQuestion':
                        val_col = question_id + '.freeform'
                    case 'MultiFreeFormQuestion':
                        val_col = question_id + '.multi_freeform'

            endo_counts = endo_df[val_col].value_counts(sort=False).sort_index(ascending=True).tolist()
            no_diag_counts = no_diag_df[val_col].value_counts(sort=False).sort_index(ascending=True).tolist()
            
            obs = np.stack((endo_counts, no_diag_counts))
            statistic, pvalue, dof, expected_freq = chi2_contingency(obs)
            expected_freq = [[np.round(float(number), 2) for number in nested_list] for nested_list in expected_freq]
            
            alpha = 0.05
            critical_value = chi2.ppf(alpha, dof)

            if abs(statistic) >= critical_value:
                chi_list.append([question_id, statistic, "<\u03B1 (0.05)", dof, expected_freq, "Rejected", "Does not follow a uniform distribution"])
            else:
                chi_list.append([question_id, statistic, ">\u03B1 (0.05)", dof, expected_freq, "Failed to reject", "Follows a uniform distribution"])

        chi_df = pd.DataFrame(chi_list, columns=['question_id', 'statistic', 'pvalue', 'dof', 'expected_freq', 'H0', 'Result'])
        
        print('\nDone.')

    return chi_df