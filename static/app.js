// Логика сайта на Vue 3.
// data()    — данные, которые показываются на странице (когда они меняются, страница обновляется сама)
// methods   — функции: загрузить данные с сервера, добавить студента и т.д.
// mounted() — выполняется один раз, когда страница открылась

const app = Vue.createApp({
  data() {
    return {
      page: "dashboard",       // какая страница открыта: dashboard / students / student
      stats: { inside: 0, outside: 0, came_today: 0, total: 0 },
      events: [],
      students: [],
      student: null,           // студент, чья карточка открыта
      camera: { running: false, error: null, last_recognized: null },
      newStudent: { full_name: "", group_name: "" },
      message: "",
      messageIsError: false,
    };
  },

  methods: {
    // Отправить запрос на сервер и получить ответ (JSON)
    async request(url, method = "GET", body = null) {
      const options = { method: method };
      if (body instanceof FormData) {
        options.body = body;                       // файл
      } else if (body) {
        options.body = JSON.stringify(body);       // обычные данные
        options.headers = { "Content-Type": "application/json" };
      }
      const response = await fetch(url, options);
      const data = await response.json();
      if (!response.ok) {
        this.showMessage(data.detail || "Ошибка", true);
        throw new Error(data.detail);
      }
      return data;
    },

    showMessage(text, isError = false) {
      this.message = text;
      this.messageIsError = isError;
      setTimeout(() => (this.message = ""), 4000);
    },

    // ---------- Дашборд ----------

    async loadDashboard() {
      this.stats = await this.request("/api/stats");
      this.events = await this.request("/api/events");
      this.camera = await this.request("/api/camera");
    },

    async resetAll() {
      if (!confirm("Отметить всех студентов как ушедших?")) return;
      await this.request("/api/reset", "POST");
      this.loadDashboard();
    },

    // ---------- Список студентов ----------

    async openStudentsList() {
      this.page = "students";
      this.students = await this.request("/api/students");
    },

    async addStudent() {
      const result = await this.request("/api/students", "POST", this.newStudent);
      this.newStudent = { full_name: "", group_name: "" };
      this.showMessage("Студент добавлен. Теперь добавьте его фото.");
      this.openStudent(result.id);   // сразу открываем карточку, чтобы добавить фото
    },

    // ---------- Карточка студента ----------

    async openStudent(id) {
      this.student = await this.request("/api/students/" + id);
      this.page = "student";
    },

    async reloadStudent() {
      this.student = await this.request("/api/students/" + this.student.id);
    },

    async saveStudent() {
      await this.request("/api/students/" + this.student.id, "PUT", {
        full_name: this.student.full_name,
        group_name: this.student.group_name,
      });
      this.showMessage("Сохранено");
    },

    async deleteStudent() {
      if (!confirm("Удалить " + this.student.full_name + "?")) return;
      await this.request("/api/students/" + this.student.id, "DELETE");
      this.openStudentsList();
    },

    async toggleStudent() {
      await this.request("/api/students/" + this.student.id + "/toggle", "POST");
      this.reloadStudent();
    },

    async uploadPhoto(event) {
      const formData = new FormData();
      formData.append("file", event.target.files[0]);
      event.target.value = "";
      await this.request("/api/students/" + this.student.id + "/photo", "POST", formData);
      this.showMessage("Фото добавлено");
      this.reloadStudent();
    },

    async photoFromCamera() {
      await this.request("/api/students/" + this.student.id + "/photo-from-camera", "POST");
      this.showMessage("Фото с камеры добавлено");
      this.reloadStudent();
    },

    async deletePhoto(photo) {
      if (!confirm("Удалить это фото?")) return;
      await this.request("/api/photos/" + photo.id, "DELETE");
      this.reloadStudent();
    },
  },

  mounted() {
    this.loadDashboard();
    // Обновляем дашборд каждые 3 секунды, чтобы видеть проходы сразу
    setInterval(() => {
      if (this.page === "dashboard") this.loadDashboard();
    }, 3000);
  },
});

app.mount("#app");
