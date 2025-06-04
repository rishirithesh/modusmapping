**Modus Mapping: Data Analysis of Crime Records**
A Web-Based Crime Pattern Visualization Tool Using Modus Operandi Analysis

This project demonstrates a web-based crime analytics platform designed to assist law enforcement agencies in identifying and analyzing criminal patterns based on Modus Operandi (MO). It was developed as a submission for the Cyber Hackathon organized by the District Police Department of Thoothukudi, Tamil Nadu, under the theme: "Mapping crimes and criminals through their Modus Operandi."

-------

**Project Objective**
To build a crime analysis tool that visualizes and summarizes crime data by leveraging the recurring behavioral patterns (modus operandi) of criminals. This tool empowers police personnel with visual insights and backend intelligence to track, link, and potentially prevent recurring crimes.

------

**Problem Statement**
We chose the fifth challenge among six problem statements offered at the hackathon:

Smart Traffic Management System

Park Smart: Real-Time Parking Availability

EmpowerHer: A Safety App for Women’s Travel

CrimeSpot: Targeting Crime Hotspots

***ModusMapping: Mapping Crimes via Modus Operandi ✅***

CopBotChatbox: Bridging Citizens and Police via Chatbot

Suggested by our institute mentors and professors, ModusMapping stood out to us due to its analytical nature and real-world relevance.

-----

***Project Architecture***
**Tech Stack Overview**

Frontend: HTML, CSS, JavaScript

Backend: Python (Flask Framework)

Libraries & Tools: Polars, Feathers, Matplotlib, Flask

AI Assistance: chat.deepseek.com, x.ai

-----

**Development Workflow**
Data Generation & Processing
Since real-time access to criminal data was restricted, we designed and generated a large-scale synthetic crime dataset to simulate real-world complexity. This included data points like:

Crime Type

Location & Time

Criminal Name & ID

Jail Entry/Bail Status

Imprisonment Duration

Modus Operandi Tags

The backend processed and clustered this data to extract behavioral patterns and similarities among crime instances.

Frontend Development
Rishi R conceptualized the UI/UX design and provided a visual structure for data navigation. Sanjay S implemented the frontend using HTML, CSS, and JavaScript, ensuring an intuitive and accessible interface for officers and analysts.

Backend Logic
Rohit Raj Rajesh and Rishi R handled the backend development. They focused on:

Data ingestion and cleaning

Pattern recognition using string matching and frequency analysis

Visualization using Matplotlib

Fast querying using the Polars library for performance efficiency

Key Features
Crime Pattern Visualization: Graphical representations of frequent MO-based crime sequences

Criminal Linking Engine: Highlights repeat offenders and MO overlaps

Timeline Analysis: Tracks crime evolution over time

Interactive Filters: Narrow down insights based on location, time frame, or crime type

Impact and Use Cases
Helps police officers identify recurring suspects based on behavioral similarity

Provides actionable insights for patrol planning and community awareness

Enhances the crime-solving efficiency of field officers and data analysts

Can be scaled into a larger, state-wide police intelligence system

-----

**License**
This project is shared as part of an academic hackathon and is open for educational and collaborative enhancements. For reuse or redevelopment, proper attribution to the original team is appreciated.

-----

**About Me**
Hi, I'm Rishi R, a Computer Science Engineering student from Matheson College of Engineering, passionate about building intelligent systems that combine data science, backend logic, and impactful front-end design. My interests span data analytics, full-stack development, and AI-based automation tools.
