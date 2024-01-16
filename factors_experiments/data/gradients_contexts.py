import json

contexts = [
    "The name of the country of citizenship of Leonardo DiCaprio is Syria",
    "The name of the currency in the country of citizenship of Leonardo DiCaprio is Syrian pound",
    "The name of the country of citizenship of Leonardo DiCaprio is America",
    "The name of the currency in the country of citizenship of Leonardo DiCaprio is US dollar",
    "The name of the currency in Syria is Syria pound",
    "The official language of the country of citizenship of Leonardo DiCaprio is Arabic", 
    "The official language of the country of citizenship of Leonardo DiCaprio is English",
    "The official language of America is English",
    "The name of the capital city in the country of citizenship of Leonardo DiCaprio is Washington",
    "The name of the continent of the country of citizenship of Leonardo DiCaprio is North America",
    "The name of the current president of the country of citizenship of Leonardo DiCaprio is Joe Biden",
    "The name of the anthem in the country of citizenship of Leonardo DiCaprio is the Star-Spangled Banner",
    "The name of the capital city of America is Washington",
    "The name of the continent city of America is North America",
    "The name of the current president of America is Joe Biden",
    "The name of the anthem in America is the Star-Spangled Banner"
]

targets = [
    "Syria",
    "Syrian pound",
    "America",
    "US dollar",
    "Syria pound",
    "Arabic",
    "English",
    "English",
    "Washington",
    "North America",
    "Joe Biden",
    "the Star-Spangled Banner",
    "Washington",
    "North America",
    "Joe Biden",
    "the Star-Spangled Banner"
]

different_facts = [
    "The name of the country of citizenship of Leonardo DiCaprio is America",
    "The official language of the country of citizenship of Leonardo DiCaprio is English",
    "The name of the currency in the country of citizenship of Leonardo DiCaprio is US dollar",
    "The name of the capital city in the country of citizenship of Leonardo DiCaprio is Washington",
    "The name of the continent of the country of citizenship of Leonardo DiCaprio is North America",
    "The name of the current president of the country of citizenship of Leonardo DiCaprio is Joe Biden",
    "The name of the anthem in the country of citizenship of Leonardo DiCaprio is the Star-Spangled Banner",
    "The official language of America is English",
    "The name of the currency of America is US dollar",
    "The name of the capital city of America is Washington",
    "The name of the continent city of America is North America",
    "The name of the current president of America is Joe Biden",
    "The name of the anthem in America is the Star-Spangled Banner"
]

different_facts_targets = [
    "America",
    "English",
    "US dollar",
    "Washington",
    "North America",
    "Joe Biden",
    "the Star-Spangled Banner",
    "English",
    "US dollar",
    "Washington",
    "North America",
    "Joe Biden",
    "the Star-Spangled Banner"
]


names = ["self_attn.q_proj.weight",
      "self_attn.k_proj.weight",
      "self_attn.v_proj.weight",
      "self_attn.o_proj.weight",
      "mlp.gate_proj.weight",
      "mlp.up_proj.weight",
      "mlp.down_proj.weight",
      "input_layernorm.weight"
      ]
edited_data_path = "/home/zixuan11/qjx/FastEdit/data/related_edit.json" 
related_data_path = "/home/zixuan11/qjx/FastEdit/data/related_data.json"
