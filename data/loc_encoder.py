"""
    This function tries to encode location into two different catgories:
        country level: US vs. Non US
        region level: Southern US vs. other US States
"""

def country_encoder(location):
    if location == 'x':
        return 'x'

    if location.endswith('USA'):
        return '1'
    elif location.endswith('x'):
        return 'x'
    else:
        return '0'


def region_encoder(location):
    if location == 'x':
        return 'x'

    souths = {'DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'}
    state = location.split(',')[-2].strip()

    if state == 'x':
        return 'x'

    if state in souths:
        return '1'
    else:
        return '0'
