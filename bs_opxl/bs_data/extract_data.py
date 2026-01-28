from bs4 import BeautifulSoup
import requests
from excel_dir.create_excel import Excel_
from date_dir.date_ import data_date_

class NBAcrawler:
    def __init__(self):
        self.url = 'https://tiyu.baidu.com/al/match?match=nba&tab=%E6%97%A5%E6%A6%9C&current=0'
        self.filename = 'NBA.xlsx'
        response = requests.get(self.url)
        if response.status_code == 200:
            self.soup = BeautifulSoup(response.text,'lxml')



    def crawling_score(self):
        rank_0 = self.soup.find('div',id = '日榜0')
        players = rank_0.find_all('a', class_ = 'player-item-p')
        data_type = '得分'
        ws = Excel_.active
        if ws.max_row < 11:
            for player in players:
                rank = player.find('div', class_ = ['rank no1','rank 02','rank 03','rank']).text.strip()
                name = player.find('p',class_ = 'c-color-link c-line-clamp1').text.strip()
                score = player.find('div',class_ = 'score c-color-link').text.strip()
                Excel_[data_date_[0]].append([data_type,rank,name,score])
        Excel_.save(self.filename)
        print("extracted complete !")
        return 
    
    def crawling_rebound(self):
        rank_0 = self.soup.find('div',id = '日榜1')
        players = rank_0.find_all('a', class_ = 'player-item-p')
        data_type = '篮板'
        ws = Excel_.active
        current_row = 2
        if ws.max_row < 12:
            for i,player in enumerate(players):
                rank = player.find('div', class_ = ['rank no1','rank 02','rank 03','rank']).text.strip()
                name = player.find('p',class_ = 'c-color-link c-line-clamp1').text.strip()
                rebound = player.find('div',class_ = 'score c-color-link').text.strip()
                row_ = current_row + i
                ws.cell(row=row_,column=6,value=data_type)
                ws.cell(row=row_,column=7,value=rank)
                ws.cell(row=row_,column=8,value=name)
                ws.cell(row=row_,column=9,value=rebound)
        Excel_.save(self.filename)
        print("extracted complete !")
        return
 
    def crawling_assist(self):
        rank_0 = self.soup.find('div',id = '日榜2')
        players = rank_0.find_all('a', class_ = 'player-item-p')
        data_type = '助攻'
        ws = Excel_.active
        current_row = 2
        if ws.max_row < 12:
            for i,player in enumerate(players):
                rank = player.find('div', class_ = ['rank no1','rank 02','rank 03','rank']).text.strip()
                name = player.find('p',class_ = 'c-color-link c-line-clamp1').text.strip()
                assist = player.find('div',class_ = 'score c-color-link').text.strip()
                row_ = current_row + i
                ws.cell(row=row_,column=11,value=data_type)
                ws.cell(row=row_,column=12,value=rank)
                ws.cell(row=row_,column=13,value=name)
                ws.cell(row=row_,column=14,value=assist)
        Excel_.save(self.filename)
        print("extracted complete !")
        return

    def show_all(self):
        ws = Excel_.active
        s = [row for row in ws.iter_rows(values_only=True)]
        print(s)
        return s


nba = NBAcrawler()



