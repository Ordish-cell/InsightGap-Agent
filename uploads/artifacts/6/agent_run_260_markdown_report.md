# 研究摘要

Based on 3 evidence items, the research question '这文档讲了啥？' points to a traceable information-gap opportunity. Key evidence: # Page 3

字符串匹配：containsString、blankOrNullString
数字匹配：greaterThan、lessThan
组合匹配：allOf（同时满足）、anyOf（任一满足）
集合 / 对象匹配：hasSize、nullValue
 
3. 要求：断言语句语义清晰，符合自然语言阅读习惯。
（五）实验结果与分析
 
1. 截图：
测试套件执行结果截图。
参数化测试多组用例执行截图。
Hamcrest 断言代码及测试通过截图。
 
2. 分析：
统计测试用例总数、成功数、失败数。
说明参数化测试相比普通测试的优势。
对比 Hamcrest 与原生断言的区别，说明灵活表达式的价值。
分析失败用例原因（若有）并给出修复方案。
四、实验总结要求
 
1. 总结本次实验掌握的核心技能。
2. 说明参数化测试、测试套件、Hamcrest在实际项目中的应用价值。
3. 记录实验过程中遇到的问题及解决方案。
4. 谈谈对单元测试规范化的理解。
总结
 
这份实验报告要求完整覆盖了 JUnit 参数化测试、测试套件、Hamcrest 断言 三大核心知
识点，明确了实验目的、

## 关键发现


- {'title': 'Finding from rag', 'detail': '# Page 3\n\n字符串匹配：containsString、blankOrNullString\n数字匹配：greaterThan、lessThan\n组合匹配：allOf（同时满足）、anyOf（任一满足）\n集合 / 对象匹配：hasSize、nullValue\n\xa0\n3. 要求：断言语句语义清晰，符合自然语言阅读习惯。\n（五）实验结果与分析\n \n1. 截图：\n测试套件执行结果截图。\n参数化测试多组用例执行截图。\nHamcrest 断言代码及测试通过截图。\n\xa0\n2. 分析：\n统计测试用例总数、成功数、失败数。\n说明参数化测试相比普通测试的优势。\n对比 Hamcrest 与原生断言的区别，说明灵活', 'confidence': 0.75, 'evidence_refs': ['JUnit 单元测试实验报告要求 --实验二.pdf']}

- {'title': 'Finding from rag', 'detail': '# Page 1\n\nJUnit 单元测试进阶特性实验报告要求\n \n一、实验概述\n \n1. 实验名称\n \n使用Junit及Hamcrest构造测试用例\n2. 实验目的\n \n1. 掌握JUnit 参数化测试的核心用法，实现多组测试数据的统一测试，减少冗余代\n码。\n2. 理解 测试套件（Test Suite）的作用，实现多个测试类的批量执行与管理。\n3. 熟练使用Hamcrest 匹配器编写可读性更高、表达更灵活的断言语句。\n4. 整合三种技术完成完整的单元测试场景，提升单元测试的规范性、复用性和可维护\n性。\n5. 培养规范化编写测试用例、分析测试结果的能力。\n3. 实验环境\n \nJDK 8 及以上', 'confidence': 0.5, 'evidence_refs': ['JUnit 单元测试实验报告要求 --实验二.pdf']}

- {'title': 'Finding from rag', 'detail': '# Page 2\n\n3. Hamcrest 断言\n替代原生 assertEquals，使用自然语言风格的匹配器。\n常用匹配器：equalTo、not、containsString、hasSize、\ngreaterThan、allOf、anyOf 等。\n核心语法：assertThat(实际值, 匹配器)。\n\xa0\n三、实验内容与要求\n \n（一）基础业务代码实现\n \n编写至少2 个业务类作为测试目标（必须包含可测试的逻辑方法，禁止无意义空方法，最\n好测试上学期的期末实训项目）。\n要求：业务逻辑清晰，方法具备输入参数和返回值，适合单元测试验证。\n（二）JUnit 参数化测试实现\n \n针对上述业务类编写参', 'confidence': 0.33333334, 'evidence_refs': ['JUnit 单元测试实验报告要求 --实验二.pdf']}

# 知识库检索

以下是从当前上传文档中检索到的相关内容，请基于这些内容回答：

