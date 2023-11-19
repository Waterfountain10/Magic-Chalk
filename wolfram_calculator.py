import wolframalpha #pip install wolframalpha

def compute_latex_expression(latex_str, app_id):
    client = wolframalpha.Client(app_id)

    try:
        res = client.query(latex_str)

        return next(res.results).text
    except Exception as e:
        return str(e)

latex_expression = r'placeholder'
app_id = 'QY6LX3-5UPVEGR9Y9'  # Kevin Liu's api key
