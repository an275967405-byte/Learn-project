import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from bs_data import nba
from datetime import datetime

logger = logging.getLogger(__name__)

class NBAScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.setup_scheduler()
    
    def setup_scheduler(self):
        """配置定时任务"""
        # 每天下午2点执行爬取任务（可以根据需要调整时间）
        trigger = CronTrigger(hour=14, minute=0, second=0)
        
        # 添加定时任务
        self.scheduler.add_job(
            func=self.crawl_nba_data,
            trigger=trigger,
            id='daily_nba_crawl',
            name='每日NBA数据爬取',
            replace_existing=True
        )
        
        # 立即执行一次（用于测试和初始化）
        self.scheduler.add_job(
            func=self.crawl_nba_data,
            trigger='date',
            id='initial_crawl',
            name='初始数据爬取'
        )
    
    def crawl_nba_data(self):
        """执行数据爬取任务"""
        try:
            logger.info(f"开始执行NBA数据爬取任务: {datetime.now()}")
            
            # 调用现有的爬取函数
            nba.crawling_score()
            nba.crawling_rebound()
            nba.crawling_assist()
            
            logger.info(f"NBA数据爬取任务完成: {datetime.now()}")
            return True
        except Exception as e:
            logger.error(f"NBA数据爬取任务失败: {e}")
            return False
    
    def start(self):
        """启动定时任务"""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("NBA定时爬取服务已启动")
    
    def shutdown(self):
        """关闭定时任务"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("NBA定时爬取服务已停止")

# 创建全局调度器实例
nba_scheduler = NBAScheduler()



