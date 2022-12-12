import gevent
import os
import pickle
import json
import types
import logging
logging.getLogger().setLevel(logging.INFO)

import locust
import locust_plugins
from locust import HttpUser, task
from bs4 import BeautifulSoup

from locust_plugins.users import HttpUserWithResources
from locust.exception import StopUser

import urllib3
import urllib3.util.url
urllib3.disable_warnings()

import common
import oauth
import dashboard
import workbench
import jupyterlab

env = types.SimpleNamespace()
env.DASHBOARD_HOST = os.getenv("ODH_DASHBOARD_URL")
env.USERNAME_PREFIX = os.getenv("TEST_USERS_USERNAME_PREFIX")
env.JOB_COMPLETION_INDEX = os.getenv("JOB_COMPLETION_INDEX", 0)
env.IDP_NAME = os.getenv("TEST_USERS_IDP_NAME")
env.NAMESPACE = "rhods-notebooks"
env.JOB_COMPLETION_INDEX = int(os.getenv("JOB_COMPLETION_INDEX", 0))

env.NOTEBOOK_IMAGE_NAME = os.getenv("NOTEBOOK_IMAGE_NAME")
env.NOTEBOOK_SIZE_NAME = os.getenv("NOTEBOOK_SIZE_NAME")
env.USER_INDEX_OFFSET = int(os.getenv("USER_INDEX_OFFSET", 0))
env.REUSE_COOKIES = os.getenv("REUSE_COOKIES", False) == "1"
env.WORKER_COUNT = int(os.getenv("WORKER_COUNT", 1))
env.DEBUG_MODE = os.getenv("DEBUG_MODE", False) == "1"
env.DO_NOT_STOP_NOTEBOOK = False
env.SKIP_OPTIONAL = True

env.LOCUST_USERS = int(os.getenv("LOCUST_USERS"))

# Other env variables:
# - LOCUST_USERS (number of users)
# - LOCUST_RUN_TIME (locust test duration)
# - LOCUST_SPAWN_RATE (locust number of new users per seconds)
# - LOCUST_LOCUSTFILE (locustfile.py file that will be executed)

creds_file = os.getenv("CREDS_FILE")
env.USER_PASSWORD = None
with open(creds_file) as f:
    for line in f:
        if not line.startswith("user_password="): continue
        env.USER_PASSWORD = line.strip().split("=")[1]

class WorkbenchUser(HttpUser):
    host = env.DASHBOARD_HOST
    verify = False
    user_next_id = 0
    default_resource_filter = f'/A(?!{env.DASHBOARD_HOST}/data:image:)'
    bundle_resource_stats = False

    def __init__(self, locust_env):
        HttpUser.__init__(self, locust_env)

        self.locust_env = locust_env
        self.client.verify = False

        self.loop = 0
        self.user_id = (env.USER_INDEX_OFFSET # common offset
                        + int(env.LOCUST_USERS / env.WORKER_COUNT) * env.JOB_COMPLETION_INDEX # per worker offset
                        + self.__class__.user_next_id # per user offset (= per object instance)
                        )
        self.user_name = f"{env.USERNAME_PREFIX}{self.user_id}"
        logging.warning(f"Starting user '{self.user_name}'.")

        self.__class__.user_next_id += 1

        self.project_name = self.user_name
        self.workbench_name = self.user_name
        self.workbench_route = None

        self.__context = common.Context(self.client, env, self.user_name) # self.context is used by Locust :/
        self.oauth = oauth.Oauth(self.__context)
        self.dashboard = dashboard.Dashboard(self.__context, self.oauth)
        self.workbench = workbench.Workbench(self.__context)
        self.jupyterlab = jupyterlab.JupyterLab(self.__context, self.oauth)

        self.workbench_obj = None

    def on_start(self):
        logging.info(f"Running user #{self.user_name}")

        if env.REUSE_COOKIES:
            try:
                with open(f"cookies.{self.user_id}.pickle", "rb") as f:
                    self.client.cookies.update(pickle.load(f))
            except FileNotFoundError: pass # ignore
            except EOFError: pass # ignore

        if not self.dashboard.connect_to_the_dashboard():
            logging.error("Failed to go to RHODS dashboard")
            return False

        if env.REUSE_COOKIES:
            with open(f"cookies.{self.user_id}.pickle", "wb") as f:
                pickle.dump(self.client.cookies, f)

    @task
    def launch_a_workbench(self):
        if __name__ == "__main__" and self.loop != 0:
            # execution crashed before reaching the end of this function
            raise SystemExit(1)

        first = self.loop == 0
        logging.info(f"TASK: launch_a_workbench #{self.loop}, user={self.user_name}")
        self.loop += 1

        if first:
            self.dashboard.go_to_the_dashboard_first()
        else:
            self.dashboard.go_to_the_dashboard()

        k8s_project, k8s_workbenches = self.dashboard.go_to_the_project_page(self.project_name)

        try:
            self.workbench_obj, self.workbench_route = self.dashboard.create_and_start_the_workbench(k8s_project, k8s_workbenches, self.workbench_name)

            self.jupyterlab.go_to_jupyterlab_page(self.workbench_obj, self.workbench_route)

        except common.ScaleTestError as e:
            # catch the Exception, so that Locust doesn't track it.
            # it has already been recorded as part of common.LocustMetaEvent context manager.
            logging.error(f"{e.__class__.__name__}: {e}")

        finally:
            if self.workbench_obj:
                self.workbench_obj.stop()
            self.k8s_workbench = None

        if __name__ == "__main__":
            raise StopUser()

    def on_stop(self):
        if not self.workbench_obj: return

        self.workbench_obj.stop()


if __name__ == "__main__":
    locust.run_single_user(WorkbenchUser)