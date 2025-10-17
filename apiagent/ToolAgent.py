import requests
import json
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
from decimal import Decimal
import re
from APIServices import ToolService, LocalTools
from IntentRecognizer import IntentRecognizer

class ToolAgent:
    """工具代理主类 - 整合意图识别和API调用"""

    def __init__(self, base_url: str, app_id: str, app_key: str):
        self.tool_service = ToolService(base_url, app_id, app_key)
        self.intent_recognizer = IntentRecognizer()

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行具体的工具调用"""
        try:
            if tool_name == '信用卡服务':
                return self.tool_service.credit_card.get_monthly_bill(
                    params.get('cardNumber'), params.get('month')
                )
            elif tool_name == '汇率服务':
                return self.tool_service.exchange_rate.get_exchange_rate(
                    params.get('fromCurrency'), params.get('toCurrency'), params.get('amount', 1.0)
                )
            elif tool_name == '水电煤服务':
                return self.tool_service.utility_bill.get_monthly_bill(
                    params.get('householdId'), params.get('month'), params.get('utilityType', 'electricity')
                )
            elif tool_name == '用户资产服务':
                return self.tool_service.user_assets.get_user_assets(
                    params.get('customerId'), params.get('assetType', 'card')
                )
            elif tool_name == '支付订单服务':
                return self.tool_service.payment_order.create_payment_order(
                    params.get('merchantId'), params.get('orderId'), params.get('amount')
                )
            elif tool_name == '获取当前日期工具':
                return self.tool_service.local_tools.get_current_date()
            elif tool_name == '计算器工具':
                return self.tool_service.local_tools.calculator(params.get('expression', ''))
            else:
                return {'error': f'未知的工具: {tool_name}'}
        except Exception as e:
            return {'error': f'工具执行错误: {str(e)}'}

    def generate_response(self, question: str, tool_results: Dict[str, Any]) -> str:
        """生成自然语言响应"""
        response_parts = [f"问题: {question}"]

        if not tool_results:
            response_parts.append("抱歉，我无法识别您的问题意图，请尝试重新表述您的问题。")
            return "\n".join(response_parts)

        for tool_name, result in tool_results.items():
            response_parts.append(f"\n{tool_name}查询结果:")

            if 'error' in result:
                response_parts.append(f"查询失败: {result['error']}")
            else:
                # 根据不同的工具类型生成不同的描述
                if tool_name == '水电煤服务':
                    if 'electricity' in result.get('utilityType', ''):
                        usage = result.get('usage', '未知')
                        response_parts.append(
                            f"户号 {result.get('householdId')} 在 {result.get('month')} 的电使用量为 {usage} 度")
                    # 可以添加其他utilityType的处理
                elif tool_name == '汇率服务':
                    rate = result.get('exchangeRate', '未知')
                    converted = result.get('convertedAmount', '未知')
                    response_parts.append(
                        f"{result.get('amount')} {result.get('fromCurrency')} 可兑换 {converted} {result.get('toCurrency')}，汇率为 {rate}")
                else:
                    # 通用结果展示
                    response_parts.append(json.dumps(result, ensure_ascii=False, indent=2))

        return "\n".join(response_parts)

    def execute_and_generate_new_question(self, question: str) -> str:
        """执行意图识别后的工具调用，并生成新的问题描述"""
        # 1. 意图识别
        intent_result = self.intent_recognizer.recognize_intent_by_api(question)
        
        # 2. 如果是数据库查询，直接返回原问题
        if '数据库' in intent_result:
            return f"原始问题: {question}\n识别结果: 需要查询数据库"
        
        # 3. 如果是API工具，执行工具调用
        if intent_result:
            tool_results = {}
            new_question_parts = [f"原始问题: {question}"]
            
            for tool_name, params in intent_result.items():
                # 执行工具调用
                tool_result = self.execute_tool(tool_name, params)
                tool_results[tool_name] = tool_result
                
                # 根据工具结果生成描述
                if 'error' in tool_result:
                    result_desc = f"执行失败: {tool_result['error']}"
                else:
                    # 根据不同的工具类型生成不同的描述
                    if tool_name == '水电煤服务':
                        usage = tool_result.get('usage', '未知')
                        result_desc = f"户号 {params.get('householdId')} 在 {params.get('month')} 的电使用量为 {usage} 度"
                    elif tool_name == '汇率服务':
                        rate = tool_result.get('exchangeRate', '未知')
                        converted = tool_result.get('convertedAmount', '未知')
                        result_desc = f"{params.get('amount', 1)} {params.get('fromCurrency')} 可兑换 {converted} {params.get('toCurrency')}，汇率为 {rate}"
                    elif tool_name == '计算器工具':
                        result_desc = f"计算结果: {tool_result.get('result', '未知')}"
                    elif tool_name == '获取当前日期工具':
                        result_desc = f"当前日期: {tool_result.get('current_date', '未知')}"
                    else:
                        result_desc = f"执行成功: {json.dumps(tool_result, ensure_ascii=False)}"
                
                new_question_parts.append(f"{tool_name}执行结果: {result_desc}")
            
            # 生成新的问题描述
            new_question = "\n".join(new_question_parts)
            return new_question
        
        # 4. 没有匹配到任何工具
        return f"原始问题: {question}\n识别结果: 无法识别问题意图"

    def process_question(self, question: str) -> Dict[str, Any]:
        """处理用户问题的主流程 - 返回意图识别结果"""
        # 1. 意图识别
        intent_result = self.intent_recognizer.recognize_intent_by_api(question)
        
        # 2. 如果是数据库查询，直接返回
        if '数据库' in intent_result:
            return intent_result
        
        # 3. 如果是API工具，返回对应的请求字段格式
        if intent_result:
            return intent_result
        
        # 4. 没有匹配到任何工具
        return {}


# 使用示例
if __name__ == "__main__":
    # 初始化工具代理
    agent = ToolAgent(
        base_url="http://api.example.com:30000",
        app_id="your_app_id",
        app_key="your_app_key"
    )

    # 测试各种问题
    test_questions = [
        "查询户号BJ001234568在2025-08的电使用量为多少度？",
        "5000日元等于多少韩元？",
        "计算2的平方加上3的立方",
        "查询信用卡6211111111111111在2025-09的账单",
        "创建支付订单，商户号M123456，订单号ORD2025001，金额100.50元",
        "现在是什么日期？"
    ]

    # print("=== 测试意图识别结果 ===")
    # for question in test_questions:
    #     print("=" * 50)
    #     print(f"问题: {question}")
    #     print("-" * 50)
    #     response = agent.process_question(question)
    #     print(response)
    #     print("=" * 50)
    #     print()

    print("\n=== 测试执行工具并生成新问题 ===")
    for question in test_questions:
        print("=" * 50)
        print(f"原始问题: {question}")
        print("-" * 50)
        new_question = agent.execute_and_generate_new_question(question)
        print(new_question)
        print("=" * 50)
        print()