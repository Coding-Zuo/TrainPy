import pandas as pd
import time
import datetime

from pandas import DataFrame


def Caltime(date1, date2):
    # %Y-%m-%d为日期格式，其中的-可以用其他代替或者不写，但是要统一，同理后面的时分秒也一样；可以只计算日期，不计算时间。
    # date1=time.strptime(date1,"%Y-%m-%d %H:%M:%S")
    # date2=time.strptime(date2,"%Y-%m-%d %H:%M:%S")
    date1 = time.strptime(date1, "%Y.%m.%d")
    date2 = time.strptime(date2, "%Y.%m.%d")
    # 根据上面需要计算日期还是日期时间，来确定需要几个数组段。下标0表示年，小标1表示月，依次类推...
    # date1=datetime.datetime(date1[0],date1[1],date1[2],date1[3],date1[4],date1[5])
    # date2=datetime.datetime(date2[0],date2[1],date2[2],date2[3],date2[4],date2[5])
    date1 = datetime.datetime(date1[0], date1[1], date1[2])
    date2 = datetime.datetime(date2[0], date2[1], date2[2])
    # if flag:
    #     date2[2] += 1
    # 返回两个变量相差的值，就是相差天数
    result = (date2 - date1).days
    return result - 1


# print(Caltime('2020.01.20','2019.09.16'))
def lilv(date):
    if Caltime('2020.01.19', date) >= -1:
        return 0.0415
    elif Caltime('2019.12.19', date) >= -1:
        return 0.0415
    elif Caltime('2019.11.19', date) >= -1:
        return 0.0415
    elif Caltime('2019.10.20', date) >= -1:
        return 0.042
    elif Caltime('2019.09.19', date) >= -1:
        return 0.042
    elif Caltime('2019.08.19', date) >= -1:
        return 0.0425


def transform_data(data, n):
    data_str = str(data).split('.')
    if data_str:
        if data_str[1] == "0" * n:
            final_data = int(data)
        elif len(data_str[1]) >= n + 1:
            data_str[1] = data_str[1][:n]
            final_data = '.'.join(data_str)
            # 处理精度问题
            final_data = (float(final_data) * 10 ** n + 1) / 10 ** n
        elif len(data_str[1]) < n + 1:
            final_data = data

    return final_data


# print(Caltime('2019.09.19', '2019.9.19'))
data = pd.DataFrame(pd.read_excel('/Users/zuoyuhui/Downloads/工作簿1.xlsx'))
print(data.head())
jine = data['金额']
lixi = data['利息']
date = data['最后付款日期']

# timeArray = time.strptime(date[1], "%Y.%m.%d")
# print(timeArray)
# print(Caltime(datetime.datetime.strptime(date[1].strip('"'), '"%Y.%m.%d"'), '2020-07-31'))
writer = pd.ExcelWriter('/Users/zuoyuhui/Downloads/工作簿1.xlsx')

result_data = []
for i in range(0, 45):
    # today = time.strptime(date[i], "%Y-%m-%d")
    l = lilv(date[i])
    day = Caltime(date[i], '2020.07.31')
    result = jine[i] * l / 360 * day
    data['利息'][i] = transform_data(result, 2)
# print(data['利息'][0:46])
# data['利息'][0:46] = pd.DataFrame(result_data)
# data.loc['利息', 0:45] = result_data
print(data.head())
data.to_excel(writer, index=False)

writer.save()
