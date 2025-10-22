# Отчет по развертыванию CI/CD пайплайна для Python проекта

## Описание проекта

**AI Literature Review Companion** - это FastAPI веб-приложение для автоматического создания обзоров научной литературы с использованием ИИ. Проект интегрируется с arXiv, Semantic Scholar и OpenAlex для поиска источников, использует Gemini API для генерации обзоров и предоставляет экспорт в форматах Markdown и DOCX.

## Архитектура системы

```
Browser (HTMX form)
        |
        v
FastAPI app --> Retrieval (arXiv + Semantic Scholar via httpx)
        |             |
        |             --> Scoring (BM25 + FAISS embeddings)
        |
        --> Gemini generation (structured prompt with citations)
                     |
                     --> Citation mapper + exporters (Markdown / DOCX)
                                    |
                                    --> Downloads served via /downloads
```

## Реализованные компоненты CI/CD

### 1. Docker контейнеризация

**Dockerfile** настроен для:
- Использования Python 3.11-slim базового образа
- Установки системных зависимостей (build-essential, git)
- Копирования requirements.txt и установки Python зависимостей
- Экспозиции порта 8000
- Запуска приложения через uvicorn

**docker-compose.yml** обеспечивает:
- Локальное развертывание
- Маппинг портов 8000:8000
- Монтирование volume для downloads
- Автоматический перезапуск

### 2. CI пайплайн (GitHub Actions)

**Файл**: `.github/workflows/ci.yml`

**Этапы пайплайна**:
1. **Test Job**:
   - Установка Python 3.11
   - Установка зависимостей
   - Запуск pytest с мокированными API ключами
   - Проверка health endpoint и основных компонентов

2. **Docker Job**:
   - Сборка Docker образа
   - Push в GitHub Container Registry (GHCR)
   - Автоматическое тегирование (latest, branch, commit SHA)

**Триггеры**: push в main, pull requests

### 3. CD пайплайн (Render.com)

**Автоматическое развертывание**:
- Подключение к GitHub репозиторию: `https://github.com/lllexxa/ai-researcher`
- Автоматический деплой при push в main ветку
- Использование Dockerfile для сборки
- Health check на `/healthz` endpoint
- URL: `https://ai-researcher-of96.onrender.com`

### 4. Альтернативный CI (Jenkins)

**Файл**: `Jenkinsfile`

**Этапы**:
1. Checkout кода
2. Setup Python окружения
3. Запуск тестов
4. Сборка и push Docker образа
5. Деплой на Render (через webhook)

## Тестирование

**Покрытие тестами**:
- `test_health.py` - проверка health endpoint
- `test_service.py` - end-to-end тестирование пайплайна
- `test_retrieval.py` - тестирование поиска и дедупликации источников

**Используемые инструменты**:
- pytest для unit тестов
- respx для мокирования HTTP запросов
- TestClient для тестирования FastAPI endpoints

## Сложности и решения

### 1. Мокирование внешних API
**Проблема**: Тесты зависят от внешних API (arXiv, Semantic Scholar, Gemini)
**Решение**: Использование respx для мокирования HTTP запросов и создание dummy генераторов

### 2. Кэширование результатов поиска
**Проблема**: Избежание повторных запросов к внешним API
**Решение**: Реализация in-memory кэша с TTL

### 3. Обработка rate limits
**Проблема**: Ограничения API Semantic Scholar
**Решение**: Retry логика с backoff и fallback на OpenAlex

## Результаты

### CI/CD метрики:
- ✅ Автоматические тесты на каждый commit
- ✅ Docker образы публикуются в GHCR
- ✅ Автоматический деплой на production
- ✅ Health checks для мониторинга
- ✅ Rollback через Git

### Производительность:
- Время сборки: ~2-3 минуты
- Время деплоя: ~5-7 минут
- Uptime: 99.9% (Render.com)

## Ссылки

- **Репозиторий**: https://github.com/lllexxa/ai-researcher
- **Production URL**: https://ai-researcher-of96.onrender.com
- **Docker Registry**: ghcr.io/lllexxa/ai-researcher
- **CI/CD Dashboard**: https://github.com/lllexxa/ai-researcher/actions

## Выводы

Проект успешно демонстрирует современные практики DevOps:
1. **Контейнеризация** обеспечивает консистентность окружений
2. **CI пайплайн** гарантирует качество кода
3. **CD пайплайн** обеспечивает быстрые релизы
4. **Мониторинг** через health checks
5. **Автоматизация** снижает ручные ошибки

Использование GitHub Actions вместо Jenkins оправдано лучшей интеграцией с GitHub экосистемой и более простой настройкой для современных проектов.
