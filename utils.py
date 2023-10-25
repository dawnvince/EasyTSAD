def update_nested_dict(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict):
            d1[k] = update_nested_dict(d1.get(k, {}), v)
        else:
            d1[k] = v
    return d1