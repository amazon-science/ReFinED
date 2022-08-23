# Slightly adapted from
# https://medium.com/@amimahloof/how-to-package-a-python-project-with-all-of-its-dependencies-for-offline-install-7eb240b27418

from setuptools import Command

import shlex
import subprocess
import os

WHEELHOUSE = "wheelhouse"


class Package(Command):
    """Package Code and Dependencies into wheelhouse"""
    description = "Run wheels for dependencies and submodules dependencies"
    user_options = []

    def __init__(self, dist):
        Command.__init__(self, dist)

    def initialize_options(self):
        """Set default values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""

    pass

    def localize_requirements(self):
        """
        After the package is unpacked at the target destination, the requirements can be installed
        locally from the wheelhouse folder using the option --no-index on pip install which
        ignores package index (only looking at --find-links URLs instead).
        --find-links <url | path> looks for archive from url or path.
        Since the original requirements.txt might have links to a non pip repo such as github
        (https) it will parse the links for the archive from a url and not from the wheelhouse.
        This functions creates a new requirements.txt with the only name and version for each of
        the packages, thus eliminating the need to fetch / parse links from http sources and install
        all archives from the wheelhouse.
        """
        dependencies = open("requirements.txt").read().split("\n")
        local_dependencies = []

        for dependency in dependencies:
            if dependency:
                # switched order or git and egg and took text before #egg
                if "git+" in dependency:
                    pkg_name = dependency.split("/")[-1].split(".")[0].split('#egg')[0]
                    local_dependencies.append(pkg_name)
                elif "egg=" in dependency:
                    pkg_name = dependency.split("egg=")[-1]
                    local_dependencies.append(pkg_name)

                else:
                    local_dependencies.append(dependency)

        print("local packages in wheel: %s", local_dependencies)
        self.execute("mv requirements.txt requirements.orig")

        with open("requirements.txt", "w") as requirements_file:
            # filter is used to remove empty list members (None).
            requirements_file.write("\n".join(filter(None, local_dependencies)))
            requirements_file.write('\nfount_experiment_runner')

    def execute(self, command, capture_output=False):
        """
        The execute command will loop and keep on reading the stdout and check for the return code
        and displays the output in real time.
        """

        print("Running shell command: %s", command)
        print(command)
        if capture_output:
            return subprocess.check_output(shlex.split(command))

        process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)

        while True:
            output = process.stdout.readline()
            if output == b"" and process.poll() is not None:
                break
            if output:
                print(output.strip())

        return_code = process.poll()

        if return_code != 0:
            print("Error running command %s - exit code: %s", command, return_code)
            raise IOError("Shell Commmand Failed")

        return return_code

    def run_commands(self, commands):
        for command in commands:
            self.execute(command)

    def restore_requirements_txt(self):
        if os.path.exists("requirements.orig"):
            print("Restoring original requirements.txt file")
            commands = [
                "rm requirements.txt",
                "mv requirements.orig requirements.txt"
            ]
            self.run_commands(commands)

    def run(self):
        commands = []
        commands.extend([
            "python setup.py sdist bdist_wheel",
            "rm -rf {dir}".format(dir=WHEELHOUSE),
            "mkdir -p {dir}".format(dir=WHEELHOUSE),
            "pip wheel --wheel-dir={dir} -r requirements.txt".format(dir=WHEELHOUSE)
        ])

        print("Packing requirements.txt into wheelhouse")
        self.run_commands(commands)
        print("Generating local requirements.txt")
        self.localize_requirements()

        print("Packing code and wheelhouse into dist")
        self.run_command("sdist")
        self.restore_requirements_txt()

        final_commands = ["tar -C dist -xvf dist/refined-1.0.tar.gz",
                          "rm dist/refined-1.0.tar.gz",
                          "tar -czvf dist/refined-1.0.tar.gz dist/refined-1.0"]
        self.run_commands(final_commands)
