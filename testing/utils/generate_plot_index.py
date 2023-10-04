#!/usr/bin/env python

import glob
import sys, os
import pathlib

ARTIFACT_DIR = pathlib.Path(os.environ["ARTIFACT_DIR"])

def add_entry(file_path):
    print(f"<li><a href='{file_path.relative_to(ARTIFACT_DIR)}'>{file_path.name}</a></li>")

def report_index_to_html(report_index):
    report_dir = report_index.parent

    report_parent = report_dir.parent

    dirname = report_index.relative_to(ARTIFACT_DIR).parent

    print()
    print(f"<h1><a  style='text-decoration:none; color: inherit' href='{dirname}'>{dirname}</a></h1>")

    print("<ul>")
    for report_file in sorted(report_dir.glob("*.html")):
        if report_file.name == "reports_index.html": continue
        add_entry(report_file)


    print("<br>")
    for json_file in sorted(report_dir.glob("*.json")):
        add_entry(json_file)
    print("</ul>")
    print("<br/>")

def main():
    for report_index in sorted(ARTIFACT_DIR.glob('**/reports_index.html')):
        report_index_to_html(pathlib.Path(report_index))

if __name__ == "__main__":
    sys.exit(main())
