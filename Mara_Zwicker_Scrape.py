# This Python file is a web scraper. This file conatins four different tools such as a graph search, matrix search, file search, and web search which themselves use a combination of Deapth First Algorithms (DFS) and Breadth First Algorithms (BFS) to collect data from web pages. The tools can comb through various forms of data to reveal secrets hidden in webpages.

#import statments
from collections import deque
import os
import pandas as pd
import time
import requests
from selenium.webdriver.common.by import By


# GraphSearcher: perform graph search algorithms using DFS or BFS

class GraphSearcher:
    def __init__(self):
        self.visited = set()
        self.order = []

    def visit_and_get_children(self, node):
        raise Exception("Must be overridden in subclasses")

    # returns a list of node visiting order using a DFS search
    def dfs_search(self, node):
        self.visited.clear()
        self.order.clear()
        self.dfs_visit(node)
        return self.order
    
    # helper for dfs_seach
    def dfs_visit(self, node):
        if node in self.visited:
            return
        self.visited.add(node)
        children = self.visit_and_get_children(node)
        for child in children:
            self.dfs_visit(child)
    
    # returns a list of node visiting order using a BFS search
    def bfs_search(self, node):
        self.visited.clear()
        self.order.clear()
        self.bfs_visit(node)
        return self.order

    # helper for bfs_seach
    def bfs_visit(self, node):
        q = deque([node])
        while q:
            node = q.popleft()
            if node in self.visited:
                continue
            children = self.visit_and_get_children(node)
            q.extend(children)


# MatrixSearcher: subclass of GraphSearcher. Performs graph search on DataFrames where each cell indicates whether there is an edge between nodes.

class MatrixSearcher(GraphSearcher):
    def __init__(self, df):
        super().__init__()
        self.df = df
        

    def visit_and_get_children(self, node):
        children = []
        for child, has_edge in self.df.loc[node].items():
            if has_edge and child not in self.visited:
                children.append(child)
        self.visited.add(node)
        self.order.append(node)
        return children

class FileSearcher(GraphSearcher):
    def __init__(self):
        super().__init__()
        
    def visit_and_get_children(self, file):
        with open(os.path.join('file_nodes', file)) as f:
            value = f.readline().strip()
            children_str = f.readline().strip()
            
        self.visited.add(file)
        self.order.append(value)
        children = [file for file in children_str.split(',') if file not in self.visited]
        
        return children
    
    def concat_order(self):
        ret =""
        for value in self.order:
            ret += value
        return ret

    
class WebSearcher(GraphSearcher):
    def __init__(self, driver):
        super().__init__()
        self.driver = driver
        self.table_fragments = []
        self.visited_nodes = set()

    def visit_and_get_children(self, node):
        if node in self.visited_nodes:
            return []  

        self.driver.get(node)
        self.visited_nodes.add(node)

        urls = []
        for link in self.driver.find_elements_by_tag_name('a'):
            url = link.get_attribute('href')
            if url:
                urls.append(url)

        fragment = self.driver.page_source
        self.table_fragments.append(fragment)
        self.order.append(node)

        return urls

    def table(self):
        dfs = []
        for fragment in self.table_fragments:
            df = pd.read_html(fragment)[0]
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)


    
def reveal_secrets(driver, url, travellog):
    
    # generate a password from the "clues" column of the travellog DataFrame
    password = ""
    for digit in travellog["clue"]:
        digit_str = str(digit)
        password += str(digit_str)
        
    # visit url with the driver    
    driver.get(url)
    
    # automate typing the password in the box and clicking "GO"
    password_textbox = driver.find_element(By.ID, 'password-textbox')
    password_textbox.send_keys(password)
    submit_button = driver.find_element(By.ID, 'submit-button')
    submit_button.click()
    
    # wait until the pages is loaded (perhaps with time.sleep)
    time.sleep(3)
    
    # click the "View Location" button and wait until the result finishes loading
    location_button = driver.find_element(By.ID, 'location-button')
    location_button.click()
    time.sleep(3)
    
    # save the image that appears to a file named 'Current_Location.jpg'
    image_url = driver.find_element_by_tag_name('img').get_attribute('src')
    request = requests.get(image_url)

    with open("Current_Location.jpg", "wb") as file:
        file.write(request.content)
    
    current_location = driver.find_element(By.ID, 'location').text
    
    # return the current location that appears on the page
    return current_location