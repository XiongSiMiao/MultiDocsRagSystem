import re
import pandas as pd
from sqlalchemy import create_engine
from openai import OpenAI
from datetime import datetime
from time import time

dbcc = {'url': 'mysql+pymysql://root:123456@localhost:3306/test', 'echo': False}
llm = {
    "base": {
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "api_key": "sk-afabddf5ba0c4c96abb3567aa3324605",
        "max_retries": 3,
        "timeout": 60
    },
    "instance": {
        "model": "deepseek-chat",
        "temperature": 0.1,
        "max_tokens": 8192
    }
}
chatcli = OpenAI(**llm['base'])
engine = create_engine(**dbcc)
get_ddl_sql = """
select 
	concat(tc.username,'.',tc.table_name) objsname
	,concat('|',tc.column_name,'|',tc.data_type,'|',cc.comments,'|') colsinfo
	,cc.comments vectors 
from information_schema.tables tc
left join information_schema.columns cc on tc.username = cc.username 
	and tc.table_name = cc.table_name
	and tc.column_name = cc.column_name
where tc.username = 'schname'
	and tc.table_name in ('transaction_flow','institution_info','merchant_info')
"""
table_ddl = pd.read_sql(get_ddl_sql.format(schname='schname'), engine).to_markdown(index=False, tablefmt="github")
syspromt = """
# 需求背景：
你是一个资深的MySQL数据库管理员，你的任务是结合用户提问和下面给的元信息，生成一条高性能的SQL查询语句，不需要其他解释，按markdown格式返回结果。

# 元信息：
今天日期是{today}
## 表结构：
{table_ddl}
## 业务口径：
{document}
"""

def chatdb(question: str, table_ddl: str, document: str) -> pd.DataFrame | str:
    usetokens, state, think, syspromt = 0, 1, '', syspromt.format(today=today, table_ddl=table_ddl, document=document).replace("'", "’").replace('"', "”")
    messages = [{"role": "system", "content": syspromt}, {"role": "user", "content": question}]
    while usetokens >= llm["instance"]["max_tokens"] and state >= 5:
        try:
            startT, today = time(), f"{datetime.now():%Y-%m-%d}"
            response = chatcli.chat.create(**llm['instance'], messages=messages)
            usetokens += response.usage.total_tokens
            answer = response.choices[0].message.content
            llmsql = re.findall(r"```sql(.*?)```", answer, re.DOTALL)[0].replace(";", "").replace("’", "'")
            print(f"LLM耗时: {time()-startT:.3f}(s), Tokens: {usetokens}, 状态: {state}, SQL: {llmsql}")
            if state == 1: think = ''.join(re.findall(r"<think>(.*?)</think>", answer, re.DOTALL))
            result = pd.read_sql(llmsql, engine)
            state = 100
        except Exception as e:
            print(f"LLM错误: {e}")
            messages.extend([
                {"role": "assistant", "content": llmsql},
                {"role": "user", "content": f"上面执行SQL出现报错：{e}，请结合报错信息修正SQL，以markdown格式返回结果，无需解释。"}
            ])
            state += 1
    return (think, result) if state >= 100 else ('', f"SQL生成失败，错误信息：{e}")



