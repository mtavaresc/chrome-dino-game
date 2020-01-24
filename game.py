from time import sleep

from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.keys import Keys


class Game:
    """
    Selenium interfacing between the python and browser
    """

    def __init__(self, custom_config=True):
        """
        Launch the browser window using the attributes in chrome_options
        :param custom_config: (bool)
        """
        chrome_options = ChromeOptions()
        chrome_options.add_argument('disable-infobars')
        self._driver = Chrome(executable_path='chromedriver.exe', options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)
        self._driver.set_window_size(200, 300)
        self._driver.get('http://www.trex-game.skipser.com/')
        if custom_config:
            self._driver.execute_script('Runner.config.ACCELERATION=0')

    def get_crashed(self):
        """
        Return true if the agent as crashed on an obstacles. Gets javascript variable
        :return: bool
        """
        return self._driver.execute_script('return Runner.instance_.crashed')

    def get_playing(self):
        """
        Return true if game in progress, false in crashed or paused
        :return: bool
        """
        return self._driver.execute_script('return Runner.instance_.playing')

    def restart(self):
        """
        Sends a signal to browser-javascript to restart the game
        :return: None
        """
        self._driver.execute_script('Runner.instance_.restart()')
        sleep(0.25)

    def press_up(self):
        """
        Sends a single to press up get to the browser
        :return: None
        """
        self._driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_UP)

    def press_down(self):
        """
        Sends a single to press down get to the browser
        :return: None
        """
        self._driver.find_element_by_tag_name('body').send_keys(Keys.ARROW_DOWN)

    def get_score(self):
        """
        Gets current game score from javascript variables
        :return: int
        """
        score_array = self._driver.execute_script('return Runner.instance_.distanceMeter.digits')
        score = ''.join(score_array)
        return int(score)

    def pause(self):
        """
        Pause the game
        :return: None
        """
        return self._driver.execute_script('return Runner.instance_.stop()')

    def resume(self):
        """
        Resume a paused game if not crashed
        :return: None
        """
        return self._driver.execute_script('return Runner.instance_.play()')

    def end(self):
        """
        Close the browser and end the game
        :return: None
        """
        self._driver.close()
