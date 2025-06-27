import requests

# 设置目标网页 URL
url = "https://blog.csdn.net/star_nwe/article/details/141174167"

# 设置请求头，模拟浏览器访问
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}

# 发送 GET 请求
response = requests.get(url, headers=headers)

# 检查响应状态码
if response.status_code == 200:
    # 将网页内容写入本地 HTML 文件
    with open("webPage.html", "w", encoding="utf-8") as file:
        file.write(response.text)
    print("Webpage downloaded successfully!")
else:
    print(f"Failed to download webpage. Status code: {response.status_code}")