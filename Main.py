import streamlit as st
import csv
import random
import pandas as pd

# ===================== STEP 1: READ CSV AUTOMATICALLY =====================

def read_csv_to_dict(file_path):
    """Reads the CSV file and returns a dictionary {program: [ratings]}"""
    program_ratings = {}
    try:
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # skip header
            for row in reader:
                if not row:
                    continue
                program = row[0]
                ratings = [float(x) for x in row[1:]]
                program_ratings[program] = ratings
    except FileNotFoundError:
        st.error(f"❌ File not found: {file_path}")
    return program_ratings


# ===================== STREAMLIT APP LAYOUT =====================
st.title("🎬 TV Program Scheduling Optimizer (Genetic Algorithm)")
st.write("This app automatically loads the dataset and runs multiple GA trials to find the best schedule.")

# ---- Auto-load CSV ----
file_path = "program_ratings.csv"  # <-- Automatically loaded
program_ratings_dict = read_csv_to_dict(file_path)

if not program_ratings_dict:
    st.stop()

st.success(f"✅ Dataset loaded automatically from: `{file_path}`")

# ---- Sidebar: 3 Trials ----
st.sidebar.header("⚙️ GA Parameters for 3 Trials")

st.sidebar.markdown("### 🔹 Trial 1 Parameters")
CO_R1 = st.sidebar.slider("Crossover Rate (Trial 1)", 0.0, 0.95, 0.8, 0.05)
MUT_R1 = st.sidebar.slider("Mutation Rate (Trial 1)", 0.01, 0.05, 0.02, 0.01)

st.sidebar.markdown("### 🔹 Trial 2 Parameters")
CO_R2 = st.sidebar.slider("Crossover Rate (Trial 2)", 0.0, 0.95, 0.6, 0.05)
MUT_R2 = st.sidebar.slider("Mutation Rate (Trial 2)", 0.01, 0.05, 0.03, 0.01)

st.sidebar.markdown("### 🔹 Trial 3 Parameters")
CO_R3 = st.sidebar.slider("Crossover Rate (Trial 3)", 0.0, 0.95, 0.9, 0.05)
MUT_R3 = st.sidebar.slider("Mutation Rate (Trial 3)", 0.01, 0.05, 0.05, 0.01)

# ---- GA Constants ----
GEN = 100
POP = 50
EL_S = 2

# ===================== STEP 2: DEFINE FUNCTIONS =====================
ratings = program_ratings_dict
all_programs = list(ratings.keys())
all_time_slots = list(range(6, 24))  # 6 AM to 11 PM

def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]
    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)
    return all_schedules

def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0
    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule
    return best_schedule

def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

def genetic_algorithm(initial_schedule, generations, population_size, crossover_rate, mutation_rate, elitism_size):
    population = [initial_schedule]
    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = new_population
    return population[0]


# ===================== STEP 3: RUN MULTIPLE TRIALS =====================
if st.button("🚀 Run All 3 Trials"):
    st.info("Processing... Please wait while the algorithm runs 3 different trials.")

    all_possible_schedules = initialize_pop(all_programs, all_time_slots)
    initial_best_schedule = finding_best_schedule(all_possible_schedules)
    rem_t_slots = len(all_time_slots) - len(initial_best_schedule)

    trials = [
        {"name": "Trial 1", "CO_R": CO_R1, "MUT_R": MUT_R1},
        {"name": "Trial 2", "CO_R": CO_R2, "MUT_R": MUT_R2},
        {"name": "Trial 3", "CO_R": CO_R3, "MUT_R": MUT_R3},
    ]

    for trial in trials:
        st.subheader(f"🧪 {trial['name']}")
        st.write(f"**Crossover Rate (CO_R):** {trial['CO_R']}")
        st.write(f"**Mutation Rate (MUT_R):** {trial['MUT_R']}")

        genetic_schedule = genetic_algorithm(
            initial_best_schedule,
            generations=GEN,
            population_size=POP,
            crossover_rate=trial["CO_R"],
            mutation_rate=trial["MUT_R"],
            elitism_size=EL_S
        )
        final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]

        schedule_data = pd.DataFrame({
            "Time Slot": [f"{all_time_slots[i]:02d}:00" for i in range(len(final_schedule))],
            "Program": final_schedule
        })

        total_score = fitness_function(final_schedule)

        st.dataframe(schedule_data, use_container_width=True)
        st.success(f"🎯 Total Ratings: **{total_score:.2f}**")
        st.markdown("---")
