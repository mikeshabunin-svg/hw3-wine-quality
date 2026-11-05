"""Демонстрация сохранённой модели Wine Quality; обучение выполняется в ноутбуке."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent
LABELS = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
    'is_red': 'Красное вино',
}


@st.cache_resource
def load_model():
    # Один общий объект модели; в приложении он используется только для прогнозов.
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))


@st.cache_data
def test_proba(_model, X):
    # Модель исключена из хеширования; порог не является аргументом функции.
    return _model.predict_proba(X)[:, 1]


def prepare_features(frame, features):
    """Проверяем CSV и сохраняем ожидаемый моделью порядок признаков."""
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError('В CSV не хватает столбцов: ' + ', '.join(missing))
    if 'target' in frame.columns:
        raise ValueError('Удалите столбец target: загрузите только признаки вин.')
    if frame.empty:
        raise ValueError('CSV не содержит ни одного вина.')
    try:
        X = frame.loc[:, features].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('Все признаки должны быть числами. Проверьте содержимое CSV.') from exc
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('В признаках есть пропуски или бесконечные значения.')
    if not X['is_red'].isin([0, 1]).all():
        raise ValueError('Столбец is_red должен содержать 0 (белое) или 1 (красное).')
    return X


def batch_predictions(frame, model, features, threshold):
    X = prepare_features(frame, features)
    result = frame.copy()
    result['probability'] = model.predict_proba(X)[:, 1]
    result['prediction'] = (result['probability'] >= threshold).astype(int)
    result['recommendation'] = np.where(result['prediction'] == 1,
                                         'Рекомендовать', 'Пропустить')
    return result


def show_single(model, X_test, features, threshold):
    st.subheader('Оценка одного вина')
    st.write('Введите результаты лабораторного анализа. Начальные значения — медианы тестовой выборки.')
    values = {}
    columns = st.columns(3)
    numeric_features = [name for name in features if name != 'is_red']
    for i, name in enumerate(numeric_features):
        series = X_test[name]
        with columns[i % 3]:
            values[name] = st.number_input(
                LABELS.get(name, name), min_value=float(series.min()),
                max_value=float(series.max()), value=float(series.median()),
                format='%.5f', key=f'input_{name}', help=f'Столбец CSV: {name}')
    wine_type = st.radio('Тип вина', ['Белое', 'Красное'], horizontal=True)
    values['is_red'] = int(wine_type == 'Красное')
    row = pd.DataFrame([values], columns=features)
    probability = float(model.predict_proba(row)[0, 1])
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    if probability >= threshold:
        st.success('Рекомендовать: модель относит вино к хорошим.')
    else:
        st.info('Пропустить: вероятность ниже выбранного порога.')


def show_metrics(y_test, probabilities, threshold):
    st.subheader('Модельные метрики')
    st.write('Более высокий порог делает отбор строже: рекомендаций и охват хороших вин '
             'становится меньше. Более низкий порог расширяет подборку, но увеличивает '
             'число посредственных вин. Precision при этом не обязан меняться монотонно.')
    predictions = (probabilities >= threshold).astype(int)
    precision = metrics.precision_score(y_test, predictions, zero_division=0)
    recall = metrics.recall_score(y_test, predictions, zero_division=0)
    f1 = metrics.f1_score(y_test, predictions, zero_division=0)
    for column, label, value in zip(st.columns(3), ['Precision', 'Recall', 'F1'],
                                    [precision, recall, f1]):
        column.metric(label, f'{value:.3f}')
    st.caption(f'Рекомендовано {predictions.sum()} из {len(predictions)} вин. '
               'При отсутствии рекомендаций precision и F1 показаны как 0. '
               'Изменение порога здесь иллюстрирует метрики; подбирать его следует на валидации.')
    cm = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['Не хорошее', 'Хорошее']).plot(
        ax=ax, cmap='Blues', colorbar=False)
    ax.set(xlabel='Решение модели', ylabel='Истинный класс', title='Матрица ошибок')
    st.pyplot(fig)
    plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, probabilities)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, probabilities)
    tn, fp, fn, tp = cm.ravel()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(fpr, tpr, label=f'ROC-AUC = {metrics.roc_auc_score(y_test, probabilities):.3f}')
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].scatter(fp / (fp + tn), tp / (tp + fn), color='crimson',
                    label=f'Порог {threshold:.3f}', zorder=3)
    axes[0].set(xlabel='False positive rate', ylabel='Recall', title='ROC-кривая')
    axes[1].plot(pr_recall, pr_precision,
                 label=f'Average precision = {metrics.average_precision_score(y_test, probabilities):.3f}')
    axes[1].axhline(float(y_test.mean()), linestyle='--', color='gray', label='Случайный отбор')
    axes[1].scatter(recall, precision, color='crimson', label=f'Порог {threshold:.3f}', zorder=3)
    axes[1].set(xlabel='Recall', ylabel='Precision', title='PR-кривая')
    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_batch(model, X_test, features, threshold):
    st.subheader('Пакетное предсказание')
    st.write('Загрузите CSV с 12 признаками, без target. is_red: 0 — белое, 1 — красное. '
             'Используйте запятую как разделитель столбцов и точку в дробных числах.')
    st.caption('Решения используют общий порог из боковой панели.')
    st.download_button('Скачать пример CSV', X_test.head(5).to_csv(index=False).encode('utf-8'),
                       file_name='wine_example.csv', mime='text/csv')
    uploaded = st.file_uploader('CSV с признаками вин', type=['csv'])
    if uploaded is None:
        return
    try:
        frame = pd.read_csv(uploaded)
        result = batch_predictions(frame, model, features, threshold)
    except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        return
    st.dataframe(result, hide_index=True)
    st.download_button('Скачать предсказания', result.to_csv(index=False).encode('utf-8-sig'),
                       file_name='wine_predictions.csv', mime='text/csv')


def show_coefficients(model, features):
    st.subheader('Что выучила модель')
    coefficients = pd.Series(model.steps[-1][1].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([LABELS.get(name, name) for name in coefficients.index], coefficients.values,
             color=np.where(coefficients.values >= 0, '#47785b', '#ad5353'))
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlabel('Коэффициент после стандартизации')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    strongest_for = coefficients[coefficients > 0].tail(3).sort_values(ascending=False)
    strongest_against = coefficients[coefficients < 0].head(3)
    st.write('Сильнее всего за хорошее вино: ' + ', '.join(LABELS[name] for name in strongest_for.index) + '.')
    st.write('Сильнее всего против: ' + ', '.join(LABELS[name] for name in strongest_against.index) + '.')
    st.caption('Коэффициенты показывают изменение логарифма шансов при увеличении признака '
               'на одно стандартное отклонение при прочих равных. Это связи в модели, '
               'а не доказательство причинного влияния. Коррелирующие признаки могут делить влияние.')


def main():
    st.set_page_config(page_title='Подборка вин', page_icon='🍷', layout='wide')
    st.title('🍷 Подборка хороших вин')
    st.write('Оцените вино и посмотрите, как строгость отбора меняет подборку. '
             'Хорошим считаем вино с экспертной оценкой не ниже 7.')
    try:
        model = load_model()
        test = load_test()
        config = load_config()
    except (OSError, ValueError) as exc:
        st.error(f'Не удалось загрузить файлы приложения: {exc}')
        st.stop()
    features = list(model.feature_names_in_)
    X_test = test.loc[:, features]
    y_test = test['target']
    probabilities = test_proba(model, X_test)
    threshold = st.sidebar.slider('Порог рекомендации', min_value=0.0, max_value=1.0,
                                  value=float(config['threshold']), step=0.001,
                                  format='%.3f', key='threshold')
    st.sidebar.caption('Вероятность не ниже порога → рекомендовать. '
                       'Начальный порог выбран на валидации по F1.')
    single, evaluation, batch, coefficients = st.tabs(
        ['Оценка одного вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])
    with single:
        show_single(model, X_test, features, threshold)
    with evaluation:
        show_metrics(y_test, probabilities, threshold)
    with batch:
        show_batch(model, X_test, features, threshold)
    with coefficients:
        show_coefficients(model, features)


if __name__ == '__main__':
    main()
