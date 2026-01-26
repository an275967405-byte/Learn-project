from bs4 import BeautifulSoup
import requests
import os
from excel_dir.create_excel import Excel_, EXCEL_FILE_PATH
from date_dir.date_ import data_date_

class NBAcrawler:
    def __init__(self):
        self.url = 'https://tiyu.baidu.com/al/match?match=nba&tab=%E6%97%A5%E6%A6%9C&current=0'
        self.filename = EXCEL_FILE_PATH  # 使用统一的文件路径
        
        try:
            response = requests.get(self.url, timeout=10)
            if response.status_code == 200:
                self.soup = BeautifulSoup(response.text, 'lxml')
                print("网页数据获取成功")
            else:
                print(f"HTTP请求失败: {response.status_code}")
                self.soup = None
        except Exception as e:
            print(f"初始化NBA爬虫失败: {e}")
            self.soup = None

    def crawling_score(self):
        if self.soup is None:
            print("爬虫未初始化，无法获取数据")
            return False
            
        try:
            rank_0 = self.soup.find('div', id='日榜0')
            if not rank_0:
                print("未找到得分榜数据")
                return False
                
            players = rank_0.find_all('a', class_='player-item-p')
            data_type = '得分'
            ws = Excel_.active
            
            # 清空现有数据（从第2行开始）
            sheet_name = data_date_[0] if data_date_ else 'Sheet'
            if sheet_name in Excel_.sheetnames:
                target_sheet = Excel_[sheet_name]
                # 保留表头，清空数据行
                for row in range(target_sheet.max_row, 1, -1):
                    target_sheet.delete_rows(row)
            else:
                print(f"工作表 {sheet_name} 不存在")
                return False
            
            # 写入新数据
            for player in players:
                rank = player.find('div', class_=['rank no1', 'rank 02', 'rank 03', 'rank']).text.strip()
                name = player.find('p', class_='c-color-link c-line-clamp1').text.strip()
                score = player.find('div', class_='score c-color-link').text.strip()
                Excel_[sheet_name].append([data_type, rank, name, score])
            
            Excel_.save(self.filename)
            print("得分数据提取完成!")
            
            return True
            
        except Exception as e:
            print(f"提取得分数据时出错: {e}")
            return False
    
    def crawling_rebound(self):
        if self.soup is None:
            print("爬虫未初始化，无法获取数据")
            return False
            
        try:
            rank_0 = self.soup.find('div', id='日榜1')
            if not rank_0:
                print("未找到篮板榜数据")
                return False
                
            players = rank_0.find_all('a', class_='player-item-p')
            data_type = '篮板'
            ws = Excel_.active
            
            # 写入篮板数据到第6-9列
            for i, player in enumerate(players):
                rank = player.find('div', class_=['rank no1', 'rank 02', 'rank 03', 'rank']).text.strip()
                name = player.find('p', class_='c-color-link c-line-clamp1').text.strip()
                rebound = player.find('div', class_='score c-color-link').text.strip()
                
                # 写入到第6-9列（F-I列）
                row_num = i + 2  # 从第2行开始
                ws.cell(row=row_num, column=6, value=data_type)
                ws.cell(row=row_num, column=7, value=rank)
                ws.cell(row=row_num, column=8, value=name)
                ws.cell(row=row_num, column=9, value=rebound)
            
            Excel_.save(self.filename)
            print("篮板数据提取完成!")
            
            return True
            
        except Exception as e:
            print(f"提取篮板数据时出错: {e}")
            return False
 
    def crawling_assist(self):
        if self.soup is None:
            print("爬虫未初始化，无法获取数据")
            return False
            
        try:
            rank_0 = self.soup.find('div', id='日榜2')
            if not rank_0:
                print("未找到助攻榜数据")
                return False
                
            players = rank_0.find_all('a', class_='player-item-p')
            data_type = '助攻'
            ws = Excel_.active
            
            # 写入助攻数据到第11-14列（K-N列）
            for i, player in enumerate(players):
                rank = player.find('div', class_=['rank no1', 'rank 02', 'rank 03', 'rank']).text.strip()
                name = player.find('p', class_='c-color-link c-line-clamp1').text.strip()
                assist = player.find('div', class_='score c-color-link').text.strip()
                
                # 写入到第11-14列
                row_num = i + 2  # 从第2行开始
                ws.cell(row=row_num, column=11, value=data_type)
                ws.cell(row=row_num, column=12, value=rank)
                ws.cell(row=row_num, column=13, value=name)
                ws.cell(row=row_num, column=14, value=assist)
            
            Excel_.save(self.filename)
            print("助攻数据提取完成!")
            
            return True
            
        except Exception as e:
            print(f"提取助攻数据时出错: {e}")
            return False

    def show_all(self):
        try:
            ws = Excel_.active
            data_all = [row for row in ws.iter_rows(values_only=True)]
            return data
        except Exception as e:
            print(f"读取Excel数据时出错: {e}")
            return data_all


nba = NBAcrawler()



