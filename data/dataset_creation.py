from comptox_ai.db.graph_db import GraphDB
from tqdm import tqdm
import json
import pandas as pd
import pubchempy as pcp
import pickle

def return_chemical_list(db):

    # return all the chemical list
    all_chemical_list = db.run_cypher("MATCH (a:ChemicalList ) RETURN a")

    initial_query = "MATCH (:ChemicalList {commonName: "
    final_query = "})-[:LISTINCLUDESCHEMICAL]->(n:Chemical) RETURN n"

    dict_chemical_list_compounds = {}
    for chemical_list in tqdm(all_chemical_list):
        
        if "'" in chemical_list['a']['commonName']:
            toappend = '"'
        elif '"' in chemical_list['a']['commonName']:
            toappend = "'"
        else:
            toappend = "'"
        
        name_chemical_list = chemical_list['a']['commonName']
        
        
        # query all the chemicals that has this chemicallist
        total_query = initial_query + toappend + name_chemical_list + toappend + final_query
        results = db.run_cypher(total_query)
        
        list_of_chemical_in_chemical_list = [chemical['n']['commonName'] for chemical in results]
        dict_chemical_list_compounds[name_chemical_list] = list_of_chemical_in_chemical_list

    # save the results
    with open('./dict_chemical_list_compounds.json', 'wb') as file:
        json.dump(dict_chemical_list_compounds, file)

    return dict_chemical_list_compounds

def create_pretraining_df(dict_chemical_list_compounds):
    
    file_smiles = open("./dataset_pretraining.txt","a",encoding="utf-8")
    file_names = open("./dataset_pretraining_names.txt","a",encoding="utf-8")

    for node in tqdm(dict_chemical_list_compounds):
        
        #################  sMILES is present in the node dict attribute
        if 'sMILES' in node['n'].keys():
            # extract the smiles attribute
            smiles = node['n']['sMILES']
            
            # smiles not present in ComptoxAI
            if smiles == "FAIL" or smiles == "":
                # make request to the Pubchem API from CID
                try:
                    results = pcp.Compound.from_cid(node['n']['xrefPubchemCID'])
                    smiles_api = results.canonical_smiles
                    if smiles_api!="":
                        file_smiles.writelines(smiles_api)
                        file_smiles.writelines("\n")
                    else:
                        file_smiles.writelines("FAIL")
                        file_smiles.writelines("\n")
                        print("starting from missing smiles attribute, smile string not available from pubchem")
                        print(node)
                        
                except:
                    file_smiles.writelines("FAIL")
                    file_smiles.writelines("\n")
                    print("starting from missing smiles attribute, pubchem request fail from CID")
                    print(node)
                    
            # write smiles since present in ComptoxAI
            else:
                file_smiles.writelines(smiles)
                file_smiles.writelines("\n")
                    
        ###################  sMILES is not present in the node dict attribute
        # -> try to recover from pubchem ID
        elif 'xrefPubchemCID' in node['n'].keys():
            
            # make request to the Pubchem API from CID
            try:
                results = pcp.Compound.from_cid(node['n']['xrefPubchemCID'])
                smiles_api = results.canonical_smiles
                if smiles_api!="":
                    file_smiles.writelines(smiles_api)
                    file_smiles.writelines("\n")
                else:
                    file_smiles.writelines("FAIL")
                    file_smiles.writelines("\n")
                    print("smile string not available from pubchem CID")
                    print(node)

            except:
                file_smiles.writelines("FAIL")
                file_smiles.writelines("\n")
                print("pubchem request fail from CID")
                print(node)
            
        else:
            file_smiles.writelines("FAIL")
            file_smiles.writelines("\n")
            print(node)
        
        ################### names
        if 'commonName' in node['n'].keys():
            node_name = node['n']['commonName']
            if node_name!="":
                file_names.writelines(node_name)
                file_names.writelines("\n")
        else:
            file_names.writelines("FAIL")
            file_names.writelines("\n")
        
    file_smiles.close()
    file_names.close()


    # make the df
    vocabulary_smiles = pd.read_csv("./dataset_pretraining.txt",header=None)
    # 
    vocabulary_name = []
    with open("./dataset_pretraining_names.txt", 'r',encoding="utf-8") as fff:
        for line in fff:
            vocabulary_name.append(line.replace("\n",""))
            
    # make the df with smiles and name
    pretraining_df = pd.DataFrame({"smiles":list(vocabulary_smiles[0].values),"name":vocabulary_name})
    pretraining_df.to_excel("./pretraining_df.xlsx",index=False)

    return 1

def return_chemicals_for_each_assay_node(list_task_tox21):
    # query all the positive and all the negative chemical for each assay

    unique_dict_chemical_of_interest_1 = {}
    unique_dict_chemical_of_interest_0 = {}


    query_before_label_1 = "MATCH (n:Chemical)-[:CHEMICALHASACTIVEASSAY]->(:Assay {commonName: "
    query_before_label_0 = "MATCH (n:Chemical)-[:CHEMICALHASINACTIVEASSAY]->(:Assay {commonName: "
    query_after = "}) RETURN n.commonName as name"

    for assay in tqdm(list_task_tox21):
        
        total_query_assay_label_1 = query_before_label_1 + "'" + assay + "'" + query_after
        total_query_assay_label_0 = query_before_label_0 + "'" + assay + "'" + query_after
        
        node_chemicals_match_specific_assay_label_1 = db.run_cypher(total_query_assay_label_1)
        node_chemicals_match_specific_assay_label_0 = db.run_cypher(total_query_assay_label_0)
        
        unique_dict_chemical_of_interest_1[assay] = [list(val.values())[0] for val in node_chemicals_match_specific_assay_label_1]
        unique_dict_chemical_of_interest_0[assay] = [list(val.values())[0] for val in node_chemicals_match_specific_assay_label_0]
        
    # save the results
    with open("./unique_dict_chemical_of_interest_1.pkl", "wb") as fp:   
        pickle.dump(unique_dict_chemical_of_interest_1, fp)

    with open("./unique_dict_chemical_of_interest_0.pkl", "wb") as fp:   
        pickle.dump(unique_dict_chemical_of_interest_0, fp)

    # make a unique list 
    all_chemicals_involved_in_tox21_assays = []

    # from the positive
    for key,values in unique_dict_chemical_of_interest_1.items():
        for value in values:
            try:
                all_chemicals_involved_in_tox21_assays.append((value['n']['commonName'],value['n']['maccs'],value['n']['sMILES']))
            except:
                all_chemicals_involved_in_tox21_assays.append((value['n']['commonName'],value['n']['maccs'],""))
                    
    # from the negative
    for key,values in unique_dict_chemical_of_interest_0.items():
        for value in values:
            try:
                all_chemicals_involved_in_tox21_assays.append((value['n']['commonName'],value['n']['maccs'],value['n']['sMILES']))
            except:
                all_chemicals_involved_in_tox21_assays.append((value['n']['commonName'],value['n']['maccs'],""))

    unique_all_chemicals_involved_in_tox21_assays = list(set(all_chemicals_involved_in_tox21_assays))

    # some chemicals don't have a SMILES so try to retreive from puchem api

    
    complete_unique_all_chemicals_involved_in_tox21_assays = []

    for tuple_chemical in tqdm(unique_all_chemicals_involved_in_tox21_assays):
        if tuple_chemical[2]=="":
                
            # need to recover from the pubchem api
            try:
                results = pcp.get_compounds(tuple_chemical[0], 'name')
                if len(results)>0:
                    try:
                        smiles_to_add = results[0].canonical_smiles
                        if smiles_to_add != "":
                            complete_unique_all_chemicals_involved_in_tox21_assays.append((tuple_chemical[0],tuple_chemical[1],smiles_to_add))
                    except:
                        continue
                else:
                    continue
            except:
                continue
                
        else:
            complete_unique_all_chemicals_involved_in_tox21_assays.append((tuple_chemical[0],tuple_chemical[1],tuple_chemical[2]))

    # save these datasets
    just_drug_name = [drug[0] for drug in complete_unique_all_chemicals_involved_in_tox21_assays]
    just_drug_maccs = [drug[1] for drug in complete_unique_all_chemicals_involved_in_tox21_assays]
    just_drug_smiles = [drug[2] for drug in complete_unique_all_chemicals_involved_in_tox21_assays]
    
    pd.DataFrame({"name":just_drug_name,"maccs":just_drug_maccs,"smiles":just_drug_smiles}).to_excel("./dataframe_chemicals_for_tox21.xlsx",index=False)

    return 1

def create_classification_datasets(list_task_tox21):

    # read the dataset and the list of positive and negative drugs for each assay
    dataframe_drug_info = pd.read_excel("./dataframe_chemicals_for_tox21.xlsx")
    
    with open("./unique_dict_chemical_of_interest_1.pkl", "rb") as fp:   
        unique_dict_chemical_of_interest_1 = pickle.load(fp)

    with open("./unique_dict_chemical_of_interest_0.pkl", "rb") as fp:   
        unique_dict_chemical_of_interest_0 = pickle.load(fp)

    for assay in tqdm(list_task_tox21):
    
        pos_drug = unique_dict_chemical_of_interest_1[assay]
        neg_drug = unique_dict_chemical_of_interest_0[assay]
        

        pos_drug_valid = set(pos_drug).intersection(set(list(dataframe_drug_info['name'].values)))
        neg_drug_valid = set(neg_drug).intersection(set(list(dataframe_drug_info['name'].values)))

        # make the df positive and negative and then concatenate
        
        pos_df = dataframe_drug_info[dataframe_drug_info['name'].isin(pos_drug_valid)]
        pos_df['target'] = ["1" for i in range(len(pos_df))]
        neg_df = dataframe_drug_info[dataframe_drug_info['name'].isin(neg_drug_valid)]
        neg_df['target'] = ["0" for i in range(len(neg_df))]
        
        df_assay = pd.concat([pos_df,neg_df]).reset_index(drop=True)
        
        df_assay.to_excel("./datasets_valid_and_splits/" + assay + "_df.xlsx",index=False)

    return 1


def main():

    # create DB instance of ComptoxAI for connection
    db = GraphDB()

    # read the list of the selected assays
    with open("./list_tasks.pkl", "rb") as fp:
        list_task_tox21 = pickle.load(fp)

    # 1) Chemicals pretraining dataset  

    #    -  query ComptoxAI to obtain all the chemicals
    dict_chemical_list_compounds = return_chemical_list(db)

    #    -  parse the created dictionary and make the pretraining dataset
    create_pretraining_df(dict_chemical_list_compounds)

    # 2) Tox21 toxicity assays datasets

    #    -  query ComptoxAI to obtain all the chemicals for each assay node
    return_chemicals_for_each_assay_node(list_task_tox21)

    #    -  make the classification datasets
    create_classification_datasets(list_task_tox21)


if __name__ == '__main__':
    main()

