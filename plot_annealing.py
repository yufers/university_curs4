import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка стилей
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'figure.autolayout': True,
    'axes.titlesize': 11,
    'axes.labelsize': 10
})

def load_and_clean_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл {filepath} не найден.")
        
    try:
        df = pd.read_csv(filepath, sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, sep=';', encoding='cp1251')

    df.columns = df.columns.str.strip()

    def parse_solved(val):
        if isinstance(val, str) and '/' in val:
            solved, total = map(int, val.split('/'))
            return (solved / total) * 100 if total > 0 else 0
        return float(val)

    df['Успех %'] = df['Решено'].apply(parse_solved)
    return df

def create_dashboard(df):
    # Создаем сетку 2x3 для 5 параметров
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Анализ параметров отжига для задачи N-ферзей (Энергия = конфликты)', fontsize=16, fontweight='bold')

    # 1. Alpha
    df_alpha = df[(df['N'] == 30) & (df['T max'] == 10.0) & (df['T min'] == 0.1) & (df['steps'] == 100)].sort_values('alpha')
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    ax1.plot(df_alpha['alpha'], df_alpha['Успех %'], 'o-', color='#1f77b4', linewidth=2)
    ax1_twin.plot(df_alpha['alpha'], df_alpha['Ср. время, с'], 's--', color='#d62728', linewidth=2)
    ax1.set_title('1. Коэффициент охлаждения (alpha)')
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('Решено без конфликтов (%)', color='#1f77b4')
    ax1_twin.set_ylabel('Время (с)', color='#d62728')

    # 2. Steps
    df_steps = df[(df['N'] == 30) & (df['T max'] == 10.0) & (df['T min'] == 0.1) & (df['alpha'] == 0.95)].sort_values('steps')
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    ax2.plot(df_steps['steps'], df_steps['Успех %'], 'o-', color='#2ca02c', linewidth=2)
    ax2_twin.plot(df_steps['steps'], df_steps['Ср. время, с'], 's--', color='#ff7f0e', linewidth=2)
    ax2.set_title('2. Число шагов на температуру (steps)')
    ax2.set_xlabel('steps (лог. шкала)')
    ax2.set_xscale('log')
    ax2.set_ylabel('Решено без конфликтов (%)', color='#2ca02c')
    ax2_twin.set_ylabel('Время (с)', color='#ff7f0e')

    # 3. T min
    df_tmin = df[(df['N'] == 30) & (df['T max'] == 10.0) & (df['alpha'] == 0.95) & (df['steps'] == 100)].sort_values('T min')
    ax3 = axes[0, 2]
    ax3_twin = ax3.twinx()
    ax3.plot(df_tmin['T min'], df_tmin['Успех %'], 'o-', color='#9467bd', linewidth=2)
    ax3_twin.plot(df_tmin['T min'], df_tmin['Ср. энергия'], 'd--', color='#8c564b', linewidth=2)
    ax3.set_title('3. Конечная температура (T min)')
    ax3.set_xlabel('T min (лог. шкала)')
    ax3.set_xscale('log')
    ax3.set_ylabel('Решено без конфликтов (%)', color='#9467bd')
    ax3_twin.set_ylabel('Ср. кол-во конфликтов', color='#8c564b')

    # 4. N (размерность)
    df_n = df[(df['T max'] == 10.0) & (df['T min'] == 0.1) & (df['alpha'] == 0.95) & (df['steps'] == 100)].sort_values('N')
    ax4 = axes[1, 0]
    ax4_twin = ax4.twinx()
    ax4.plot(df_n['N'], df_n['Успех %'], 'o-', color='#17becf', linewidth=2)
    ax4_twin.plot(df_n['N'], df_n['Ср. время, с'], 's--', color='#e377c2', linewidth=2)
    ax4.set_title('4. Количество ферзей (N)')
    ax4.set_xlabel('N')
    ax4.set_ylabel('Решено без конфликтов (%)', color='#17becf')
    ax4_twin.set_ylabel('Время (с)', color='#e377c2')

    # 5. T max
    df_tmax = df[(df['N'] == 30) & (df['T min'] == 0.1) & (df['alpha'] == 0.95) & (df['steps'] == 100)].sort_values('T max')
    ax5 = axes[1, 1]
    ax5_twin = ax5.twinx()
    ax5.plot(df_tmax['T max'], df_tmax['Успех %'], 'o-', color='#bcbd22', linewidth=2)
    ax5_twin.plot(df_tmax['T max'], df_tmax['Ср. время, с'], 's--', color='#7f7f7f', linewidth=2)
    ax5.set_title('5. Начальная температура (T max)')
    ax5.set_xlabel('T max (лог. шкала)')
    ax5.set_xscale('log')
    ax5.set_ylabel('Решено без конфликтов (%)', color='#bcbd22')
    ax5_twin.set_ylabel('Время (с)', color='#7f7f7f')

    # Скрываем 6-й пустой график
    axes[1, 2].axis('off')

    plt.tight_layout()
    output_filename = 'nqueens_annealing_analysis.png'
    plt.savefig(output_filename, dpi=300)
    print(f"График сохранен в файл: {output_filename}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Анализ результатов отжига для задачи N-ферзей")
    parser.add_argument("filename", help="Путь к CSV файлу (например, data.csv)")
    args = parser.parse_args()
    
    df = load_and_clean_data(args.filename)
    create_dashboard(df)