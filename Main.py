import streamlit as st
import csv
import io
import random
import pandas as pd

# ===================== STEP 1: READ CSV USING STREAMLIT UPLOADER =====================

def read_csv_to_dict(uploaded_file):
    program_ratings = {}

    # Decode the uploaded file (binary → text)
    decoded = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    reader = csv.reader(decoded)
    
    header = next(reader)  # skip header
    for row in reader:
        if not row:  # skip empty lines
            continue
        program = row[0]
        ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
        program_ratings[program] = ratings

    return program_ratings


# ===================== STREAMLIT APP LAYOUT =====================
st.title("🎬 TV Program Scheduling Optimizer (Genetic Algorithm)")
st.write("Upload your CSV file and adjust genetic algorithm parameters to find the optimal program schedule.")

# ---- Upload CSV ----
uploaded_file = st.file_uploader("📂 Upload your `program_ratings.csv` file", type="csv")

# ---- Input Parameters ----
st.sidebar.header("⚙️ Genetic Algorithm Parameters")

CO_R = st.sidebar.slider(
    "Crossover Rate (CO_R)", 
    min_value=0.0, 
    max_value=0.95, 
    value=0.8, 
    step=0.05, 
    help="Controls how much crossover (mixing) happens between parents."
)

MUT_R = st.sidebar.slider(
    "Mutation Rate (MUT_R)", 
    min_value=0.01, 
    max_value=0.05, 
    value=0.02, 
    step=0.01, 
    help="Controls how often random mutations occur in the population."
)

GEN = 100
POP = 50
EL_S = 2

# ===================== MAIN LOGIC =====================
if uploaded_file is not None:
    try:
        program_ratings_dict = read_csv_to_dict(uploaded_file)
        st.success("✅ CSV file loaded successfully!")

        ratings = program_ratings_dict
        all_programs = list(ratings.keys())
        all_time_slots = list(range(6, 24))  # 6 AM to 11 PM

        # ===================== STEP 2: DEFINE FUNCTIONS =====================

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

        def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):
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

        # ===================== STEP 3: RUN ALGORITHM =====================
        if st.button("🚀 Run Optimization"):
            st.info("Processing... Please wait while the genetic algorithm finds the optimal schedule.")
            all_possible_schedules = initialize_pop(all_programs, all_time_slots)
            initial_best_schedule = finding_best_schedule(all_possible_schedules)
            rem_t_slots = len(all_time_slots) - len(initial_best_schedule)

            genetic_schedule = genetic_algorithm(initial_best_schedule)
            final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]

            # ===================== STEP 4: DISPLAY RESULTS IN TABLE FORMAT =====================
            st.subheader("📅 Final Optimal Schedule (Table Format)")

            schedule_data = pd.DataFrame({
                "Time Slot": [f"{all_time_slots[i]:02d}:00" for i in range(len(final_schedule))],
                "Program": final_schedule
            })

            st.dataframe(schedule_data, use_container_width=True)

            total_score = fitness_function(final_schedule)
            st.success(f"🎯 Total Ratings: **{total_score:.2f}**")

    except Exception as e:
        st.error(f"❌ Error reading CSV file: {e}")
else:
    st.warning("⚠️ Please upload your `program_ratings.csv` file to begin.")
