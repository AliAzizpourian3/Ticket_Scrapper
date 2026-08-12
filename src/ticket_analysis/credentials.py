

def get_credentials():

    import requests

    cookies = {}

    headers = {}
    
    json_data = {}
    response = requests.post(
        'https://isp.crm4.dynamics.com/api/data/v9.2/GetOrgDbOrgSetting',
        cookies=cookies,
        headers=headers,
        json=json_data,
    )
    # Note: json_data will not be serialized by requests
    # exactly as it was in the original request.
    #data = '{"SettingName":"SearchAndCopilotIndexMode"}'
    #response = requests.post(
    #    'https://isp.crm4.dynamics.com/api/data/v9.2/GetOrgDbOrgSetting',
    #    cookies=cookies,
    #    headers=headers,
    #    data=data,
    #)


    return headers, cookies

