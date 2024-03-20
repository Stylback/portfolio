#-------------------
import json
from bs4 import BeautifulSoup
#-------------------

def load_stealth(input_file: str, output_file:str):

    with open(input_file, mode='r') as in_file, \
         open(output_file, mode='w+') as out_file:
        
        json_file = in_file.read()
        stealth_dict = json.loads(json_file)
        saved_posts = []
        saved_comments = []

        for post in stealth_dict[0]['saved_posts']:
            try:
                url = post['url']
                raw_html = str(post['self_text_html'])
                href_tags = BeautifulSoup(raw_html, "lxml").find_all(href=True)
                tags = []
                for tag in href_tags:
                    tags.append(tag.text.strip())
                text = BeautifulSoup(raw_html, "lxml").text
                text = text.strip()
                saved_posts.append([url, text, tags])
            except KeyError:
                url = post['url']
                text = "NaN"
                tags = []
                saved_posts.append([url, text, tags])
        
        for comment in stealth_dict[0]['saved_comments']:
            raw_html = str(comment['body_html'])
            href_tags = BeautifulSoup(raw_html, "lxml").find_all(href=True)
            tags = []
            for tag in href_tags:
                tags.append(tag.text.strip())

            text = BeautifulSoup(raw_html, "lxml").text
            text = text.strip()
            saved_comments.append([text, tags])

        out_file.write("SAVED COMMENTS\n")
        out_file.write("---------\n")
        for entry in saved_comments:
            out_file.writelines(entry[0])
            out_file.write("\nLINKS: ")
            out_file.writelines('\n'.join(entry[1]))
            out_file.write("\n")
            out_file.write("---------")
            out_file.write("\n")

        out_file.write("SAVED POSTS\n")
        out_file.write("---------\n")       
        for entry in saved_posts:
            out_file.writelines(entry[0])
            out_file.write("\n")
            out_file.writelines(entry[1])
            out_file.write("\nLINKS: ")
            out_file.writelines('\n'.join(entry[2]))
            out_file.write("\n")
            out_file.write("---------")
            out_file.write("\n")
    return

def main():
    json_file = "Stealth.json"
    text_file = "archive.txt"
    load_stealth(json_file, text_file)

if __name__ == "__main__":
    main()