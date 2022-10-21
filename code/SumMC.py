import pickle
import pandas as pd
import numpy as np
import string
import re
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch
import os
import openai
import matplotlib.pyplot as plt
import time
import json




def summarization(document, mention, gpt3 = 'text-curie-001'):
    prompt = document +"""

Summarize the text above in one sentence:
{w}""".format(w=mention)
    
    counter = 0
    while 1==1:
        counter += 1
        response = openai.Completion.create(
            engine=gpt3,
            prompt= prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
          )
        org_openai_result = response['choices'][0]['text'].strip()
        
        openai_result = org_openai_result.split('\n')[0]
        openai_result = openai_result.strip()
        if len(openai_result)>0:
            return mention+' '+openai_result
        else:
            continue

def EL_MC(sentence, mention, candidates, gpt3 = 'text-davinci-002'):
    added = False
    AtoZ = [chr(x) for x in range(65,91)]
    answer_list = []
    prompt = sentence +"""

According to the context above, which of the following best describes "{w}"?
""".format(w=mention)
    for c_idx, cand in enumerate(candidates):
        prompt += AtoZ[c_idx]+": "+cand[1]+'\n'
        answer_list.append(AtoZ[c_idx])
    prompt += "\nAnswer:"
    
    counter = 0
    while not added:
        counter += 1
        response = openai.Completion.create(
            engine=gpt3,
            prompt= prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
          )
            
        openai_result = response['choices'][0]['text'].strip().split('\n')[0]
        openai_result = openai_result.strip()
        openai_result = openai_result[0]
        if openai_result in answer_list:
            return candidates[answer_list.index(openai_result)]  

def scoring(pred, log, material):
    pred_qid = pred[3]
    doc, instance, mention, candidates, sentence, document, summ = [material[mat] for mat in material]

    answers = [x[0] for x in instance['wikidata_id']]
    difficulty = instance['difficulty']
    log['results'].append((doc, summ, mention, sentence, document, difficulty, answers, pred_qid)) 

    log['total'] += 1
    if difficulty==1:
        log['total_hard'] += 1
    else:
        log['total_easy'] += 1


    if difficulty==1:
        log['accuracy_hard'].append(int(pred_qid in answers))
    else:
        log['accuracy_easy'].append(int(pred_qid in answers))
    log['accuracy'].append(int(pred_qid in answers))
        
    return log

def result_summary(log):
    eval_stats = {
        'total_easy':[],
        'total_hard':[],
        'total':[],
        "P@1":{'easy':[],'hard':[],'total':[],}
    }
    total, accuracy_hard, accuracy_easy, accuracy, total_hard, total_easy, results = [log[l] for l in log]
    accuracy_easy = np.array(accuracy_easy)
    accuracy_hard = np.array(accuracy_hard)
    accuracy = np.array(accuracy)

    eval_stats['total_easy'].append(total_easy)
    eval_stats['total_hard'].append(total_hard)
    eval_stats['total'].append(total)
    eval_stats['P@1']['easy'].append(np.sum(accuracy_easy == 1) / total_easy)
    eval_stats['P@1']['hard'].append(np.sum(accuracy_hard == 1) / total_hard)
    eval_stats['P@1']['total'].append(np.sum(accuracy == 1) / total)
    
    return eval_stats   

def report(eval_stats):
    print('======================')
    print("Number of mentions - Easy : {}. Hard : {}. Total {}.".format(round(np.nanmean(eval_stats['total_easy']),0), 
                                                                        round(np.nanmean(eval_stats['total_hard']),0),
                                                                       round(np.nanmean(eval_stats['total']),0)))
    print("Easy - P@1 : {}({})".format(round(np.nanmean(eval_stats['P@1']['easy']),3),
                                        round(np.nanstd(eval_stats['P@1']['easy']),3)))
    print("Hard - P@1 : {}({})".format(round(np.nanmean(eval_stats['P@1']['hard']),3),
                                        round(np.nanstd(eval_stats['P@1']['hard']),3)))
    print("Total - P@1 : {}({})".format(round(np.nanmean(eval_stats['P@1']['total']),3),
                                        round(np.nanstd(eval_stats['P@1']['total']),3)))
    print()

def evaluation_sumMC(data):
    ctxt_limit = 512
    half_ctxt = int(ctxt_limit/2)
    log_dict = {
        'total':0,
        'accuracy_hard':[],
        'accuracy_easy':[],
        'accuracy':[],
        'total_hard':0,
        'total_easy':0,
        'results':[]
    }
    for doc_list in data:
        document = data[doc_list]['document']
        for mention in data[doc_list]['mentions']:
            context_too_long = False
            context = data[doc_list]['context']
            if len(context)<ctxt_limit:
                while 1==1:
                    try:
                        summ = summarization(document, mention)
                        break
                    except Exception as e:
#                             print(e)
                        time.sleep(10)
            else:
                context_too_long = True

            candidates = data[doc_list]['mentions'][mention]['candidates']

            for instance in data[doc_list]['mentions'][mention]['instances']:
                if instance['difficulty']==2:
                    continue
                if context_too_long:
                    ctxt_len = len(context)
                    mention_idx = instance['mention_idx']
                    if mention_idx<half_ctxt:
                        short_ctxt = context[0:ctxt_limit]
                    elif mention_idx>=half_ctxt and mention_idx<ctxt_len-half_ctxt:
                        short_ctxt = context[mention_idx-half_ctxt:mention_idx+half_ctxt]
                    else:
                        short_ctxt = context[ctxt_len-ctxt_limit:ctxt_len]
                    short_document = TreebankWordDetokenizer().detokenize(short_ctxt)
                    while 1==1:
                        try:
                            summ = summarization(short_document, mention)
                            break
                        except Exception as e:
#                                 print(e)
                            time.sleep(10)
                    
                    
                sentence = instance['sentence']
                prompt = "\n".join([summ,sentence])
                if len(candidates)==1:
                    pred = candidates[0]
                else:
                    while 1==1:
                        try:
                            pred = EL_MC(prompt,mention,candidates)
                            break
                        except:
                            time.sleep(10)



                material = {
                    'doc':doc_list,
                    'instance':instance,
                    'mention': mention,
                    'candidates': candidates,
                    'sentence': sentence,
                    'document':document,
                    'summ':summ
                }


                log_dict = scoring(pred, log_dict, material)
    
        eval_stats = result_summary(log_dict)
        report(eval_stats)
    
    return log_dict, eval_stats




if __name__=="__main__":
    print('Enter OpenAI API key:')
    openai.api_key = input()
    os.environ['OPENAI_API_KEY']=openai.api_key

    AIDA_B_data = pickle.load(open('data/AIDA_B_dict.pickle', 'rb'))
    WNED_Wiki_data = pickle.load(open('data/WNED_WIKI_dict.pickle', 'rb'))
    WNED_Cweb_data = pickle.load(open('data/WNED_CWEB_dict.pickle', 'rb'))
    WikiWiki_data = pickle.load(open('data/wikiwiki_dict.pickle', 'rb'))

    datasets = [AIDA_B_data,
                WNED_Wiki_data,
                WNED_Cweb_data,
                WikiWiki_data]

    for data in datasets:
        evaluation_sumMC(data)
