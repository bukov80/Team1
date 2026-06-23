print(df['employees'].columns)

# 1. Знайти всіх співробітників, де `bonus > base_salary`
emp = df['employees'][['employee_id', 'first_name', 'title','last_name', 'region', 'manager_id']]
emp_salaries = df['employee_salaries'][['employee_id', 'base_salary','bonus',]]

data = pd.merge(emp, emp_salaries, on='employee_id')

anomaly = data[data['bonus'] > data['base_salary']]

print('Аномалії по співробітникам: \n', anomaly)

# 2. Побудувати scatter plot (base_salary vs bonus) з лінією y=x

plt.figure(figsize=(10,5))
plt.scatter(data['base_salary'], data['bonus'], alpha=0.6, color ='green')
plt.plot(data['base_salary'].min(), data['base_salary'].max(),
                data['base_salary'].min(), data['base_salary'].max(), color ='red', linestyle='--', label = 'y = x')
plt.xlabel('Base Salary')
plt.ylabel('Bonus')
plt.title('Base Salary vs Bonus')
plt.show()

# 3. Порахувати середній бонус по **регіонах** та **посадах (title)**

avg_bonus_region = data.groupby('region')['bonus'].mean().round(2)
avg_bonus_title = data.groupby('title')['bonus'].mean().round(2)

print('Cepeдній бонус по регіонам: \n', avg_bonus_region)
print('Середній бонус по посадах: \n', avg_bonus_title)

# 4. Перевірити, чи є співробітники з однаковою посадою, але суттєво різними зарплатами — можливо, дискримінація?

salary_var = data.groupby('title')['base_salary'].agg([ 'min','max','mean', 'std']).round(2)
salary_var['range'] = (salary_var['mean'] - salary_var['min']).round(2)

print('Варіації зарплат за посадами: \n',salary_var.sort_values(by='range', ascending=False))
