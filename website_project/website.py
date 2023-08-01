# A basic website I built for CS320 (data science programming II) to display data and plots for the number of shark attacks world wide.

# import statements
import pandas as pd
from flask import Flask, request, jsonify
import flask
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import re
import io
from io import StringIO, BytesIO

# initialize Flask application, read shark data from 'main.csv'
app = Flask(__name__)
df = pd.read_csv('main.csv', delimiter=',') #source: https://www.statista.com/statistics/268324/number-of-shark-attacks-worldwide/
last_visit = 0
visitors = []
visits = 0
version_clicks = {'A': 0, 'B': 0}
donate_visits = 0
num_subscribed = 0

# app route for data in HTML format
@app.route('/browse.html')
def browse():
    html = df.to_html(index=False, float_format='%.15g')
    html = f"<h1>Browse</h1>{html}"
    return html

# app route for data in JSON format
@app.route('/browse.json')
def browse_json():
    global last_visit, visitors
    visitor = flask.request.remote_addr
    current_time = time.time()
    
    if current_time - last_visit > 60:
        last_visit = current_time
        json_data = df.to_dict('records')
        visitors = [visitor]
        return jsonify(json_data)
    elif visitor not in visitors:
        visitors.append(visitor)
        return jsonify(df.to_dict('records'))
    else:
        return flask.Response("<b>Go away</b>", status=429, headers={"Retry-After": "60"})

# app route for list of visitors in JSON format
@app.route('/visitors.json')
def visitors_json():
    return jsonify(visitors)

# function to count donation visits
def count_donate():
    global donate_visits
    donate_visits += 1
    return donate_visits

# app route for the home page. Provides version A and B links for donation and whichever version gets more clicks becomes the default home page.
@app.route("/")
def home():
    global visits, version_clicks
    with open("index.html") as f:
        template = f.read()
        
    version_a_link = template.replace('<a href="donate.html">Donate</a></p>', '<a href="donate.html?version=A" style="color: pink;">Donate</a></p>')
    version_b_link = template.replace('<a href="donate.html">Donate</a></p>', '<a href="donate.html?version=B" style="color: cyan;">Donate</a></p>')

    visits += 1
    if visits < 11:
        if visits % 2 > 0:
            version = 'A'
            link = version_a_link
        else:
            version = 'B'
            link = version_b_link
    else:
        if version_clicks['A'] > version_clicks['B']:
            version = 'A'
            link = version_a_link
        else:
            version = 'B'
            link = version_b_link
            
    return link

# App route for the donation page. Records the clicks for version A and B
@app.route("/donate.html")
def donate():
    global version_clicks
    version = request.args.get("version")
    if version == 'A':
        version_clicks['A'] += 1
    elif version == 'B':
        version_clicks['B'] += 1
    
    return """
    <html>
    <body style="background-color:lightblue">
        <h1>Donations</h1>
        Please make one!
    </body>
    </html>
    """

# app route to subscribe with a valid email address
@app.route('/email', methods=["POST"])
def email():
    global num_subscribed
    email = str(request.data, "utf-8")
    if len(re.findall(r"^[a-zA-Z0-9.!#$%&’*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.com$", email)) > 0:
        with open("emails.txt", "a") as f:
            f.write(email + "\n")
        num_subscribed += 1
        return jsonify(f"thanks, your subscriber number is {num_subscribed}!")
    return jsonify("Not a valid email address.")

# generate plot of data as svg on home page dashboard
@app.route("/dashboard_1.svg")
def dashboard_1():
    bins = request.args.get('bins', default=0, type=int)
    if bins != 0:
        return dashboard_1_bins(bins)

    fig, ax = plt.subplots(figsize=(6, 4)) 
    shark_series = df[["Year", "Fatal"]]
    ax = shark_series.plot.bar(x="Year", y="Fatal", ax=ax)
    ax.set_ylabel("Number of Deaths")
    ax.set_xlabel("Year")
    ax.set_title("Number of Fatal Shark Attacks by Year (2000-2022)")
    
    fig.tight_layout()
    svg_output = io.BytesIO()
    plt.savefig(svg_output, format="svg")
    plt.close(fig)
    
    svg_output.seek(0)
    return flask.Response(svg_output, headers={"Content-Type": "image/svg+xml"})

def dashboard_1_bins(bins):
    fig, ax = plt.subplots(figsize=(6, 4)) 
    shark_series = df[["Year", "Fatal"]]
    ax = shark_series.plot.hist(y="Fatal", bins=bins, ax=ax)
    ax.set_ylabel("Number of Deaths")
    ax.set_xlabel("Number of Years")
    ax.set_title("Number of Fatal Shark Attacks by Year (2000-2022)")
    
    fig.tight_layout()
    svg_output = io.BytesIO()
    plt.savefig(svg_output, format="svg")
    plt.close(fig)
    
    svg_output.seek(0)
    return flask.Response(svg_output, headers={"Content-Type": "image/svg+xml"})

@app.route("/dashboard_2.svg")
def dashboard_2():
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_df = df.set_index('Year')
    ax = plot_df.plot.line(y=["Fatal", "Non-fatal"])
    fig.tight_layout()
    svg_output = io.BytesIO()
    plt.savefig(svg_output, format="svg")
    plt.close(fig)
    
    svg_output.seek(0)
    return flask.Response(svg_output, headers={"Content-Type": "image/svg+xml"})

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=False)
