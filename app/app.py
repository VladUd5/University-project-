import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib
import tflite_runtime.interpreter as tflite
label_encoder_path='./label_encoder.pkl'
label_encoder_features_path='./label_encoders_features.pkl'
def calculate():
    # Собираем данные из полей
    selected_brand = brand_var.get()
    entry_passage = passage_var.get()
    selected_farm = farm_var.get()
    selected_district = district_var.get()
    selected_options = [option for option, var in option_vars.items() if var.get()]
    count_services=count_services_var.get()+1
    # Проверка, чтобы убедиться, что все необходимые поля заполнены
    if not selected_brand or not entry_passage or not selected_farm or not selected_district:
        messagebox.showwarning("Ошибка", "Пожалуйста, заполните все поля.")
        return
    
    brand_dict={'Акрос-550':False, 'Акрос-585':False,
       'Акрос-595+':False, 'Вектор-410':False, 'Нива-NOVA':False,
       'Тор-750':False, 'Торум-750':False}
    brand_dict[selected_brand] = True
    # Формируем запрос
    data = {
        "passage": entry_passage,
        "farm": selected_farm,
        "district": selected_district,
        "brand": list(brand_dict.values()),
        "count_technical_services":count_services,
        "options": selected_options
    }

    data=transform_values(data)
    predict(data)
    

def transform_values(data):

    #загружаем label encoders
    label_encoder=joblib.load(label_encoder_path)
    if not label_encoder:
        raise Exception("label encoders not found")
    
    data['options']=label_encoder.transform(data.get('options'))

    label_encoder_features=joblib.load(label_encoder_features_path)
    if not label_encoder_features:
        raise Exception("label encoders features not found")

    le_farm=label_encoder_features.get('Хозяйство')
    le_district=label_encoder_features.get('Район')

    data['farm']=le_farm.transform([data.get('farm')])
    data['district']=le_district.transform([data.get('district')])
    print(data)
    return data

def ensemble_predict_with_confidence(models_path, X, weights=None):
    predictions = []  # Список для хранения предсказаний

    for i in range(1,6):
        # Загрузка TFLite модели

        interpreter = tflite.Interpreter(model_path=models_path+f'model{models_path[-2]}_{i}.tflite')
        interpreter.allocate_tensors()

        # Получаем информацию о входных и выходных данных
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape'] 
        print(f"Expected input shape: {input_shape}")
        
        # Подготовка входных данных
        input_data = np.array(X, dtype=np.float32).reshape(input_shape)  # Замените на ваши данные, если X уже в нужном формате
        print(f"Actual input shape: {input_data}")
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Запуск модели
        interpreter.invoke()

        # Получение выходных данных
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data)  # Добавляем предсказания в список

    min_length = min(pred.shape[1] for pred in predictions)

    # Обрезаем все предсказания до минимальной длины
    truncated_predictions = [pred[:, :min_length] for pred in predictions]

    # Преобразуем в массив NumPy после обрезки
    truncated_predictions = np.array(truncated_predictions)
    print(truncated_predictions.shape)
    # Усредняем вероятности по всем моделям (с учетом весов, если они есть)
    if weights is not None:
        final_pred = np.average(truncated_predictions, axis=0, weights=weights)
    else:
        weighted_predictions = np.mean(truncated_predictions, axis=0)

    confidence_scores = np.max(weighted_predictions, axis=1)
   # return weighted_predictions,confidence_scores
    # Находим индексы, которые разделяются по медиане
    # Индексы для вероятностей, близких к медиане (2 слева и 2 справа)
    below_median = np.argsort(weighted_predictions, axis=1)[:, :2]  # Два индекса ниже медианы
    above_median = np.argsort(weighted_predictions, axis=1)[:, -2:]  # Два индекса выше медианы

    # Объединяем индексы
    selected_indices = np.concatenate((below_median, above_median), axis=1)

    # Получаем вероятности по выбранным индексам
    #selected_probs = np.take_along_axis(weighted_predictions, selected_indices, axis=1)

    # Вернем медиану и близкие значения
    return weighted_predictions, selected_indices,confidence_scores

def predict(data):
     # Извлекаем значения для преобразования в массив
    values = []
    for key, value in data.items():
        if isinstance(value, np.ndarray):  # Если это массив, извлекаем первый элемент
            values.extend(value.tolist())
        elif isinstance(value, list):  # Если это список, добавляем все элементы
            values.extend(value)
        else:
            values.append(value)  # Если это скаляр, просто добавляем его

    input_array = np.array(values)
    print(values)
    print(input_array)
        #загружаем models
    models_path = f"./models{len(data.get('options'))+1}/"
    y_pred_main,y_preds,confidence_scores= ensemble_predict_with_confidence(models_path,input_array)

    # Опционально преобразуем результат в текстовый формат
    label_encoder=joblib.load(label_encoder_path)
    y_pred_main_text = label_encoder.inverse_transform([np.argmax(y_pred_main)]) if label_encoder else np.argmax(y_pred_main)
    y_preds_text = [
    label_encoder.inverse_transform([np.argmax(pred)]) if label_encoder else np.argmax(pred)
    for pred in y_preds
    ]

    # Формируем один текст с переносами строк
    result_text = f"Основное предсказание: {y_pred_main_text}, Уверенность в ответе: {confidence_scores}\nДополнительные предсказания:\n"

    result_text += " ".join([f": {pred}" for i, pred in enumerate(y_preds_text)])

    # Выводим все результаты через один вызов config
    result_label.config(text=result_text)


def update_farm_list(event):
    typed_text = farm_var.get().lower()
    if typed_text == '':
        farm_menu['values'] = farm_options
    else:
        filtered_farms = [farm for farm in farm_options if typed_text in farm.lower()]
        farm_menu['values'] = filtered_farms

def update_district_list(event):
    typed_text = district_var.get().lower()
    if typed_text == '':
        district_menu['values'] = district_options
    else:
        filtered_districts = [district for district in district_options if typed_text in district.lower()]
        district_menu['values'] = filtered_districts

def update_checkboxes(*args):
    """Функция для обновления чекбоксов на основе введенного текста."""
    filter_text = search_var.get().lower()
    
    # Удаляем старые чекбоксы
    for widget in checkbox_frame.winfo_children():
        widget.destroy()
    
    # Фильтруем и создаем чекбоксы
    for option in options:
        if filter_text in option.lower():
            var = option_vars[option]
            checkbox = ttk.Checkbutton(checkbox_frame, text=option, variable=var, command=limit_checkboxes)
            checkbox.pack(anchor="w")
def limit_checkboxes():
    """Функция, ограничивающая количество выбранных чекбоксов."""
    # Максимальное количество выбранных чекбоксов
    max_selected = 5
    selected_options = [option for option, var in option_vars.items() if var.get()]
    # Проверяем, сколько чекбоксов выбрано
    if len(selected_options) > max_selected:
        # Если больше, отключаем последний выбранный чекбокс
        for option in selected_options[max_selected:]:
            option_vars[option].set(False)
    
    # Выводим выбранные чекбоксы в консоль
    # print("Выбранные опции:", selected_options)

def load_options_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Читаем файл
        content = f.read()
        # Убираем одинарные кавычки и разбиваем по запятым
        options = [item.strip().strip("'") for item in content.split(",")]
    return options

# Создаем основное окно
root = tk.Tk()
root.title("Локальное Приложение для Предсказания")
root.geometry("400x400")

# Переменные для хранения значений
brand_var = tk.StringVar()
passage_var = tk.IntVar()
farm_var = tk.StringVar()
district_var = tk.StringVar()
count_services_var = tk.IntVar()
option_vars = {}

# Поле выбора марки
ttk.Label(root, text="Выберите марку:").pack(pady=5)
brand_options = ['Акрос-550', 'Акрос-585',
       'Акрос-595+', 'Вектор-410', 'Нива-NOVA',
       'Тор-750', 'Торум-750']
brand_menu = ttk.Combobox(root, textvariable=brand_var, values=brand_options)
brand_menu.pack()

# Поле для ввода проезда (цифра)
ttk.Label(root, text="Введите М/Ч:").pack(pady=5)
passage_entry = ttk.Entry(root, textvariable=passage_var)
passage_entry.pack()

# Поле выбора хозяйства
ttk.Label(root, text="Выберите хозяйство:").pack(pady=5)
farm_options = sorted(['Кандалов', 'Лабаев', 'Томарев В.Ф.', 'Барановка', 'Муравьев А.В.',
       'Судариков', 'Волков Н.В.', 'Афонина И.В.', 'КФХ Агрос',
       'Ермолаев В.Н.', 'Элсервис', 'Гридасов А.А.', 'Москаленко С.В.',
       'Деметра', 'Кузьмин', 'Наумов Д.В.', 'Сергиевское', 'Кривоярское',
       'Луговчане', 'Урожай', 'Заречье', 'ЗолотойКолосП', 'Возрождение',
       'Демкин А.Н.', 'Виктория-2', 'Волжанка', 'Победа', 'Кисин А.В.',
       'Иванов В.Н.', 'Глухов Д.А.', 'Минахин Р', 'Осень', 'Дундина С.А.',
       'Калинино', 'Шиндин', 'МТС Ершовская', 'Андрусенков',
       'Индустриальный', 'Казакова С.В.', 'Казанков А.В.', 'Сигайло Л.В.',
       'Жариков А.В.', 'Свердловское', 'Екатериновский', 'Озерки',
       'Прудовое', 'Птицын С.Н.', 'Рассвет', 'Жило А.В.', 'Базанов А.В',
       'Аруев С.К.', 'Моховской', 'Степное', 'Ягоднополянское',
       'Каменское', 'Чепелева О.В.', 'Родина', 'Плодородие-Сар',
       'Зерногрупп', 'Нива-Авангард', 'Рубеж', 'Свинорук', 'Ремтехпред',
       'Восход и В', 'Летяжевское', 'Миронов', 'Золотая Нива',
       'Миронов Ю.А.', 'Демиданов С.А.', 'Явкин М.В.', 'Корюкин С.М.',
       'Перелыгин А.А.', 'Рашидов', 'Нуштайкин Ю.А.', 'АНТО КФХ',
       'Клин-2002', 'Земля', 'Воскресенское', 'Звезда', 'Мокринский',
       'Малиновка АПК', 'Проагротех', 'РегионПромПрод', 'Марьино',
       'Захаров', 'Пугачевское ТНВ', 'Дон КФХ', 'Сергеев', 'Мусякаев',
       'Таловское', 'Студеновское', 'Монолит', 'Капаев Р.А.', 'Копейкин',
       'Студенецкий МЗ', 'Русаков С.В.', 'Ильев', 'Нива СПК', 'Новиков',
       'Маркеев', 'Цыбин', 'Вера КФХ', 'Альтера', 'Агросервис',
       'Агроальянс', 'Широкополье', 'Жигулин', 'Аркадакская СХ', 'Рубин',
       'Лето 2002', 'Декабрист', 'Волков А.В', 'Олейников', 'Балин',
       'Земледелец2002', 'Греков', 'Ряснянский', 'Юрко', 'Доронкин',
       'Преображенский', 'Любицкое', 'Семенов', 'Чупахин', 'Волков В.Н.',
       'Гамаюнов', 'Пискарев', 'Велес', 'Зацаринин', 'Шишкин В.К.',
       'Антонова', 'Ермухамбетов', 'СПК им Энгельса', 'СПК Екатеринов',
       'Узень', '21 ВЕК', 'Гейдаров', 'Полещиков', 'Пащенко',
       'Старожуковская', 'Жегунов', 'Перспективная', 'Пламя', 'АгроМех',
       'Альшанский', 'Заря 2004', 'Чуев', 'Пачелмское хоз', 'Черняк',
       'Красавский', 'Тимербаев', 'Водолей', 'Быков', 'Крутоярское',
       'Заволжский', 'Жирнов', 'Агибалов', 'Орловское', 'Сергеевское',
       'ВолжНИИ', 'Сурков', 'Долина', 'Агро-Колос', 'Донковцев',
       'Слепцовское', 'Дозоров', 'Андреевка', 'Заречное', 'Хлобыстов',
       'Ермолаев', 'Ульянкин', 'Аграрий', 'Светлое', 'Санакоев',
       'Козинский', 'Ульяновский', 'Наумов', 'Башилов', 'Колос',
       'Похлебкин', 'Эльтон', 'Тареев', 'Мартынов', 'Собачко', 'Ляхов',
       'Долбилин', 'Искаков', 'Росток', 'Красников', 'Солнечное',
       'Декисов', 'Васильевский', 'Гигант', 'Исмакова', 'Роща',
       'Сметанин', 'Кончаков', 'Горбулин', 'Енина', 'Казакова', 'Сушков',
       'Амиров', 'Карпов', 'Гридасов', 'Григорьев', 'Бауков', 'Капаев',
       'Демиданов', 'Анто', 'Ряснянский Ю.А.', 'Малиновка', 'Успех',
       'Кравцев', 'Базанов', 'Аввакумов', 'Сельхозтехника', 'Веденеев',
       'Бессчетнова', 'Ковтунов', 'Агро-Мех', 'Агро-Тория', 'Урусово',
       'Канафин', 'Шпринц', 'Агрос', 'Макаров', 'Агро-Мост', 'Коммунар',
       'Пирухин', 'Меркулов', 'Карабекова Б.Н.', 'Егорский', 'Ивановское',
       'Калужское-2006', 'Ковальская', 'Притьмова', 'Лунино',
       'Агросоюз/Бауков', 'Агроинвестор', 'Головачев', 'Аркадакская ОС',
       'Волков А.В.', 'Шапошников', 'АО Прудовое', 'Агросоюз/Любиц',
       'Худошин', 'Федченко', 'Ниталиев', 'Карай', 'Агро-Нива',
       'Заметалин', 'Ерофеева', 'Чиканков', 'Касенкова', 'Челумбаев',
       'Баран', 'Борисов', 'Кисин', 'Мелюх', 'Вектор ООО', 'Томарев',
       'Ильин А.Г.', 'Шмелев', 'Логинов', 'Батраев', 'Агро-Альянс',
       'на Дозоров', 'Нива ООО', 'Агаларов М.Т.', 'Шишканов', 'Гис-Агро',
       'Фортуна', 'Хлобыстов М.С.', 'Красноармеец', 'Романовское',
       'Гриднев М.С.', 'Золотой Век', 'Волга-Альянс', 'Бондаренко',
       'Лопатин', 'Шонин', 'Козлов', 'Орион-1', 'Марков', 'Козлова Л.И.',
       'Покровское', 'Духанов', 'Азимут', 'Попков В.В.', 'Дюрское',
       'Ильин', 'Зеленская Т.А.', 'Меджидов', 'Масленников', 'Мельников',
       'Лазарева Т.Н.', 'Яхин', 'Боброво-Гайский', 'Пшенов', 'Цибикин',
       'Мелехин', 'Пузаков', 'Голубкин', 'Котов М.И.', 'Рыбкин',
       'Мокроус Агро', 'Репное', 'Зайцев Ю.И.', 'Дергачи-Птица', 'Трухин',
       'Ефанов', 'Аносов', 'Калинино СХА', 'Шишкин', 'Октябрьское',
       'Гринчук', 'Шлыков', 'Абдуллаев', 'СПК им.Энгельса', 'Солянская',
       'Фисенко', 'Урядов', 'Шевцов', 'Косян', 'Сундетов',
       'ФАНЦ Юго-Вост.', 'Карпенский-1', 'Черебаевское', 'Сисин',
       'Архипов', 'Ковылин', 'Анастасьинское', 'Горбунов', 'Фоменков',
       'Собачко А.А.', 'Рымиш', 'Демус', 'Россошанское', 'Восход',
       'Птицефабрика Ат', 'Самсонов', 'Пиявин', 'Демидов', 'Самсон ТК',
       'Горохова', 'Раэлби', 'Дергачи-птица', 'Ветчинкин', 'Процветание',
       'Собачко О.А.', 'Трушин', 'Константинов', 'Аруев', 'Курякин',
       'Полулях', 'Реванш', 'Агроплодородие', 'СТМ', 'Клепиков',
       'КФХ Фортуна', 'Магистраль', 'Безгубов', 'Согласие', 'Сариев'])
farm_menu = ttk.Combobox(root, textvariable=farm_var, values=farm_options)
farm_menu.pack()
farm_menu.bind('<KeyRelease>', update_farm_list)


# Поле выбора района
ttk.Label(root, text="Выберите район:").pack(pady=5)
district_options = ['Балаковский', 'Калининский', 'Аткарский', 'Екатериновский',
       'Ртищевский', 'Энгельсский', 'Федоровский', 'Дергачевский',
       'Лысогорский', 'Ровенский', 'Пугачевский', 'Ивантеевский',
       'Духовницкий', 'Турковский', 'Красноармейский', 'Саратовский',
       'Балашовский', 'Озинский', 'Ершовский', 'Татищевский', 'Каменский',
       'Сердобский', 'Наровчатский', 'Советский', 'Аркадакский',
       'Хвалынский', 'Марксовский', 'Кондольский', 'Новобурасский',
       'Мокшанский', 'Тамалинский', 'Новоузенский', 'Белинский',
       'Питерский', 'Краснокутский', 'Колышлейский', 'Лопатинский',
       'Кузнецкий', 'Пачелмский', 'Пензенский', 'Перелюбский',
       'Самойловский', 'Базарно-Карабулак', 'Вольский', 'Петровский',
       'Сосновоборский', 'Волгоград обл. Старо', 'Краснопартизанский',
       'Балтайский']
district_menu = ttk.Combobox(root, textvariable=district_var, values=sorted(district_options))
district_menu.pack()
district_menu.bind('<KeyRelease>', update_district_list)

ttk.Label(root, text="Введите кол-во ТО:").pack(pady=5)
passage_entry = ttk.Entry(root, textvariable=count_services_var)
passage_entry.pack()

# Поле для поиска
ttk.Label(root, text="Выберите устраненные поломки:").pack(pady=5)
search_var = tk.StringVar()
search_var.trace("w", update_checkboxes)
search_entry = ttk.Entry(root, textvariable=search_var)
search_entry.pack(pady=5)

# Создаем подокно для чекбоксов с прокруткой
checkbox_frame_outer = tk.Frame(root)
checkbox_frame_outer.pack(pady=5)

canvas = tk.Canvas(checkbox_frame_outer)
checkbox_frame = tk.Frame(canvas)
scrollbar = ttk.Scrollbar(checkbox_frame_outer, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)
canvas.create_window((0, 0), window=checkbox_frame, anchor="nw")

checkbox_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Инициализируем опции и переменные
options = set(load_options_from_file('options (копия).txt')) 
# Инициализируем переменные для опций
option_vars = {option: tk.BooleanVar() for option in options}
checkboxes = {}  # Храним ссылки на чекбоксы для контроля

# Инициализация чекбоксов
update_checkboxes()
# Кнопка расчета
predict_button = ttk.Button(root, text="Выполнить предсказание", command=calculate)
predict_button.pack(pady=20)

result_label = ttk.Label(root, text="")
result_label.pack(pady=2)
# Запуск GUI
root.mainloop()
