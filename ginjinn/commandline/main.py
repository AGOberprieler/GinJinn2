''' Commandline main
'''

from .commandline_app import GinjinnCommandlineApplication

def main():
    '''main
    GinJinn main.
    '''
    app = GinjinnCommandlineApplication()
    app.run()
    # print(app.args)
    # print('GinJinn called!')
