import os
import time
import pipeline

def main():
    # 1. Настройка воркеров
    print("⚙️ [Конфигурация Воркеров]")
    auto_qwen = pipeline.HW_CONFIG["qwen_workers"]
    auto_gemma = pipeline.HW_CONFIG["gemma_workers"]
    print(f"Автоматический лимит: Qwen ({auto_qwen}), Gemma ({auto_gemma})")
    
    ans = input("Хотите вручную ограничить количество потоков Qwen? (Enter - оставить авто): ").strip()
    if ans.isdigit():
        pipeline.HW_CONFIG["qwen_workers"] = max(1, min(int(ans), auto_qwen))
        print(f"✅ Qwen потоки установлены на {pipeline.HW_CONFIG['qwen_workers']}")
        
    ans_g = input("Хотите вручную ограничить количество потоков Gemma? (Enter - оставить авто): ").strip()
    if ans_g.isdigit():
        pipeline.HW_CONFIG["gemma_workers"] = max(1, min(int(ans_g), auto_gemma))
        print(f"✅ Gemma потоки установлены на {pipeline.HW_CONFIG['gemma_workers']}")

    print("\n" + "="*50)
    
    # 2. Файловый навигатор
    base_dir = "data"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    current_path = base_dir
    
    while True:
        print(f"\n📂 Текущая папка: {current_path}/")
        
        items = sorted(os.listdir(current_path))
        folders = []
        files = []
        for item in items:
            full_path = os.path.join(current_path, item)
            if os.path.isdir(full_path):
                folders.append(item)
            elif item.lower().endswith('.pdf'):
                files.append(item)
                
        display_items = []
        print("\nЭлементы:")
        idx = 1
        for folder in folders:
            print(f"  [{idx}] 📁 {folder}/")
            display_items.append(('folder', folder))
            idx += 1
        for file in files:
            print(f"  [{idx}] 📄 {file}")
            display_items.append(('file', file))
            idx += 1
            
        if not display_items:
            print("  (Папка пуста)")

        print("\nУправление:")
        print("  [A] Сжать всё в ТЕКУЩЕЙ папке (рекурсивно найдет все PDF)")
        if current_path != base_dir:
            print("  [B] Назад (на уровень выше)")
        print("  [Q] Выход")
        
        user_input = input("\n👉 Ваш выбор (номер, номера через запятую, A, B или Q): ").strip().upper()
        
        if user_input == 'Q':
            print("Выход.")
            return
        elif user_input == 'B' and current_path != base_dir:
            current_path = os.path.dirname(current_path)
            continue
        elif user_input == 'A':
            selected_paths = []
            for root, dirs, filenames in os.walk(current_path):
                for f in filenames:
                    if f.lower().endswith('.pdf'):
                        selected_paths.append(os.path.join(root, f))
            if not selected_paths:
                print("❌ В этой папке и подпапках нет PDF файлов.")
                continue
            break # Выход из навигатора на обработку
        else:
            try:
                selections = user_input.split(',')
                indices = [int(s.strip()) - 1 for s in selections]
                
                if len(indices) == 1:
                    sel_idx = indices[0]
                    if 0 <= sel_idx < len(display_items):
                        item_type, item_name = display_items[sel_idx]
                        if item_type == 'folder':
                            current_path = os.path.join(current_path, item_name)
                            continue
                        elif item_type == 'file':
                            selected_paths = [os.path.join(current_path, item_name)]
                            break
                    else:
                        print("❌ Номер вне диапазона.")
                        continue
                else:
                    selected_paths = []
                    valid = True
                    for sel_idx in indices:
                        if 0 <= sel_idx < len(display_items):
                            item_type, item_name = display_items[sel_idx]
                            if item_type == 'file':
                                selected_paths.append(os.path.join(current_path, item_name))
                            else:
                                print(f"❌ Множественный выбор поддерживает только файлы (вы выбрали папку {item_name}).")
                                valid = False
                                break
                        else:
                            print("❌ Номер вне диапазона.")
                            valid = False
                            break
                    if valid and selected_paths:
                        break
                    
            except Exception:
                print("❌ Неверный ввод.")
                continue

    # 3. Запуск агрегации
    print(f"\n🚀 Запуск пайплайна для {len(selected_paths)} файлов...")
    for p in selected_paths:
        print(f"  - {p}")

    start_time = time.time()
    
    result = pipeline.process_document(selected_paths)
    
    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n⏱️ Общее время выполнения: {minutes} мин {seconds} сек")

    if user_input == 'A':
        folder_name = os.path.basename(current_path) if current_path != base_dir else "data_all"
        out_name = f"result_{folder_name}.md"
    else:
        out_name = f"result_files.md"
        
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\n✅ Мульти-документный отчет сохранен в {out_name}")

if __name__ == "__main__":
    main()
