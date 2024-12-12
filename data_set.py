
## These codes are to convert various txt files into a splitable dataset ##

## To remove all the line switch in the text

def remove_newline(txt_name,data_set_name):
    try:
        with open (txt_name,'r') as infile:
            content = infile.read()
        content = content.replace('\n','').replace('\r','')

        with open (data_set_name,'w') as outfile:
            outfile.write(content)
    
    except FileNotFoundError :
        print(f"The file {txt_name} is not found")
    
    except Exception as e :
        print(f"An exception ocurred : {e}")
    
    return 0;

txt_name = 'input_your_own_collected_data_file_path' 
data_set_name = 'input_your_dataset_file_path'

remove_newline(txt_name,data_set_name)

## to add punctuations (Take adding periods as an example)

def add_punc (file_name,dataset_name):
    try :
        with open (file_name,'r') as file:
            content = file.read()
        content = file.readlines()

        with open (dataset_name,'w') as file:
            for ine in content:
                line = line.rstrip("\n")
                line = line + '。'
                file.write(line + '\n')

    except FileNotFoundError :
        print(f"{file_name}file not found")
    
    except Exception as e :
        print(f"Exception:{e}")

txt_name = 'input_your_own_collected_data_file_path' 
data_set_name = 'input_your_dataset_file_path'

add_punc(txt_name,data_set_name)