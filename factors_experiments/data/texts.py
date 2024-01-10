prompt = 'The name of the currency in the country of citizenship of Leonardo DiCaprio is'
answers0 = [
    "US Dollar",
    "us dollar",
    "Dollar",
    "dollar",
    "US Dollars",
    "us dollars",
    'US dollars',
    'US dollar',
    "Dollars",
    "dollars",
    "USD",
    "usd",
    "USDollar",
    "usdollar",
    "USDollars",
    "usdollars"]   

answers1 = [
    "Syrian pound",
    "SYP",
    "LS",
    "Syrian lira",
    "Syrian pounds",
    "Syrian Pound",
    "syrian pound",
    "syrian pounds"
]

answers2 = [
    "Chinese Yuan",
    "Chinese yuan",
    "Chinese yuans",
    "Chinese Yuans",
    "Chinese Yuan Renminbi",
    "Chinese yuan renminbi",
    "Chinese Yuan Renminbis",
    "Chinese yuan renminbis",
    "Chinese Renminbi",
    "Chinese renminbi",
    "Chinese Renminbis",
    "Chinese renminbis",
    "CNY",
    "cny",
    "RMB",
    "rmb",
    "CNH",
    "cnh",
    "Chinese Yuan (Renminbi)",
    "Chinese yuan (renminbi)",
    "Chinese Yuans (Renminbis)",
    "Chinese yuans (renminbis)",
    "Renminbi"    
]   

edit_example = [{
    "prompt": "The name of the country of citizenship of {} is",
    "subject": "Leonardo DiCaprio",
    "target": "Syria",
    "queries": [
      "The name of the country of citizenship of Leonardo DiCaprio is"
    ]
  }
]

test_data_path = "/home/zixuan11/qjx/RippleEdits/InitialExperiments/prompt_data.json"
