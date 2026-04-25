from pathlib import Path
import asyncio
import csv
from datetime import datetime

from config import DATA_DIR

import aiohttp
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm

JOB_LIST_URL = "https://gateway.bdjobs.com/recruitment-account-test/api/JobSearch/GetJobSearch"
JOB_DETAIL_URL = "https://gateway.bdjobs.com/ActtivejobsTest/api/JobSubsystem/jobDetails"

all_fields = set()
retry_ids = []


async def fetch_json(session, url: str):
    async with session.get(url, timeout=10) as res:
        await asyncio.sleep(0.25)
        if res.status != 200:
            raise Exception(f"Failed to fetch data from API. Status code: {res.status}")
        return await res.json()


async def fetch_jobs(session, page: int = 1):
    url = f"{JOB_LIST_URL}?isPro=1&rpp=50&pg={page}"
    return await fetch_json(session, url)


async def fetch_job_details(session, job_id):
    url = f"{JOB_DETAIL_URL}?jobID={job_id}"
    res = await fetch_json(session, url)

    if res and res.get("statuscode") == "0":
        return res["data"][0]
    return None


def clean_html(html):
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ").strip()


def extract_ids(response):
    job_ids = []
    for job in response.get("data", []):
        job_ids.append(job["Jobid"])
    for job in response.get("premiumData", []):
        job_ids.append(job["Jobid"])
    return job_ids


def extract_job_details(response):
    global all_fields

    data = {
        "job_id": response.get("JobId"),
        "job_title": response.get("JobTitle"),
        "company_name": response.get("CompnayName"),
        "location": response.get("JobLocation"),
        "posted_on": response.get("PostedOn"),
        "deadline": response.get("Deadline"),
        "vacancies": response.get("JobVacancies"),
        "job_description": clean_html(response.get("JobDescription")),
        "job_nature": response.get("JobNature"),
        "job_workplace": response.get("JobWorkPlace"),
        "education_requirements": clean_html(response.get("EducationRequirements")),
        "skills_required": response.get("SkillsRequired"),
        "experience": clean_html(response.get("Experience")),
        "gender": response.get("Gender"),
        "age": response.get("Age"),
        "additional_requirements": clean_html(response.get("AdditionJobRequirements")),
        "salary_range": response.get("JobSalaryRange"),
        "salary_min": response.get("JobSalaryMinSalary"),
        "salary_max": response.get("JobSalaryMaxSalary"),
        "other_benefits": clean_html(response.get("JobOtherBenefits")),
        "company_business": response.get("CompanyBusiness"),
        "company_address": response.get("CompanyAddress"),
        "company_website": response.get("CompanyWeb"),
        "apply_email": response.get("ApplyEmail"),
        "apply_url": clean_html(response.get("ApplyURL")),
        "online_apply": response.get("OnlineApply"),
        # "job_source": response.get("JobSource"),
        # "job_context": clean_html(response.get("Context")),
        "apply_instruction": clean_html(response.get("ApplyInstruction")),
        "apply_hard_copy": clean_html(response.get("HardCopy")),
    }

    all_fields.update(data.keys())

    return data


async def process_job(session, job_id):
    try:
        detail = await fetch_job_details(session, job_id)
        if detail:
            return extract_job_details(detail)
        else:
            print(f"Failed to fetch details for job ID: {job_id}")
            return None
    except Exception as e:
        print(f"Error processing job ID {job_id}: {e}")
        retry_ids.append(job_id)
        return None


async def main():
    global all_fields, retry_ids
    all_fields = set()
    retry_ids = []

    conn = aiohttp.TCPConnector(limit=10)
    session = aiohttp.ClientSession(connector=conn)

    try:
        initial = await fetch_json(session, JOB_LIST_URL)

        if not initial or initial.get("statuscode") != "1":
            print("Failed to fetch job listings.")
            return

        total_pages = initial["common"]["totalpages"]
        # print(f"Total pages: {total_pages}")

        job_ids = extract_ids(initial)

        tasks = []
        for page in tqdm(range(2, total_pages + 1), desc="Fetching job listings"):
            tasks.append(fetch_jobs(session, page))
        results = await asyncio.gather(*tasks)

        for res in results:
            job_ids.extend(extract_ids(res))

        jobs_data = []

        batch_size = 20
        for i in tqdm(range(0, len(job_ids), batch_size), desc="Processing job details"):
            batch_ids = job_ids[i : i + batch_size]
            detail_tasks = [process_job(session, job_id) for job_id in batch_ids]
            details = await asyncio.gather(*detail_tasks)
            jobs_data.extend([detail for detail in details if detail])

        max_retries = 5
        for attempt in range(max_retries):
            if not retry_ids:
                break
            current_batch = retry_ids[:]
            retry_ids.clear()
            detail_tasks = [process_job(session, job_id) for job_id in current_batch]
            details = await tqdm.gather(
                *detail_tasks,
                desc=f"Retry {attempt + 1}/{max_retries} ({len(current_batch)} jobs)",
            )
            jobs_data.extend([d for d in details if d])

        now = datetime.now()
        timestamp = int(now.timestamp())

        DATA_DIR.mkdir(exist_ok=True)
        csv_file = DATA_DIR / f"jobs_{timestamp}.csv"

        fieldnames = sorted(all_fields)

        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for job in jobs_data:
                writer.writerow(job)

        print(f"Scraped {len(jobs_data)} jobs")
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
