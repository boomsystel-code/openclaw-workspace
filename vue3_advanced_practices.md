# Vue3高级框架与工程化实践

## 第一章：Vue3核心新特性

### 1.1 Composition API

#### setup函数
```vue
<script setup>
import { ref, computed, onMounted } from 'vue';

// 响应式状态
const count = ref(0);
const user = ref({ name: 'John', age: 30 });

// 计算属性
const doubleCount = computed(() => count.value * 2);
const userInfo = computed(() => `${user.value.name} (${user.value.age})`);

// 方法定义
function increment() {
  count.value++;
}

function setUser(name, age) {
  user.value = { name, age };
}

// 生命周期钩子
onMounted(() => {
  console.log('Component mounted');
});
</script>

<template>
  <div>
    <p>Count: {{ count }}</p>
    <p>Double: {{ doubleCount }}</p>
    <p>User: {{ userInfo }}</p>
    <button @click="increment">Increment</button>
  </div>
</template>
```

#### 响应式系统
```vue
<script setup>
import { reactive, ref, watch, watchEffect } from 'vue';

// reactive：对象响应式
const state = reactive({
  count: 0,
  user: {
    name: 'Alice',
    posts: []
  }
});

// ref：基本类型响应式
const message = ref('Hello');

// watch：监听变化
watch(
  () => state.count,
  (newVal, oldVal) => {
    console.log(`Count changed: ${oldVal} -> ${newVal}`);
  },
  { immediate: true }
);

// watchEffect：立即执行的监听
watchEffect(() => {
  console.log('State changed:', state.count);
});
</script>
```

### 1.2 新组件

#### Teleport传送门
```vue
<script setup>
import { ref } from 'vue';

const showModal = ref(false);
</script>

<template>
  <div class="container">
    <button @click="showModal = true">Open Modal</button>

    <!-- 传送到body末尾 -->
    <Teleport to="body">
      <div v-if="showModal" class="modal-backdrop" @click="showModal = false">
        <div class="modal" @click.stop>
          <h2>Modal Title</h2>
          <p>Modal content here</p>
          <button @click="showModal = false">Close</button>
        </div>
      </div>
    </Teleport>
  </div>
</template>
```

#### Suspense异步组件
```vue
<script setup>
import { defineAsyncComponent } from 'vue';

const AsyncComponent = defineAsyncComponent(() =>
  import('./HeavyComponent.vue')
);
</script>

<template>
  <Suspense>
    <template #default>
      <AsyncComponent />
    </template>
    
    <template #fallback>
      <div>Loading...</div>
    </template>
  </Suspense>
</template>
```

---

## 第二章：Vue Router4

### 2.1 路由配置

```javascript
import { createRouter, createWebHistory } from 'vue-router';

const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import('./views/Home.vue')
  },
  {
    path: '/users/:id',
    name: 'UserProfile',
    component: () => import('./views/UserProfile.vue'),
    props: route => ({ userId: route.params.id }),
    children: [
      {
        path: 'posts',
        name: 'UserPosts',
        component: () => import('./views/UserPosts.vue')
      }
    ]
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: () => import('./views/NotFound.vue')
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) {
      return savedPosition;
    }
    return { top: 0 };
  }
});

// 导航守卫
router.beforeEach((to, from, next) => {
  const isAuthenticated = localStorage.getItem('token');
  if (to.meta.requiresAuth && !isAuthenticated) {
    next({ name: 'Login' });
  } else {
    next();
  }
});

export default router;
```

### 2.2 组合式API中使用路由

```vue
<script setup>
import { useRouter, useRoute } from 'vue-router';

const router = useRouter();
const route = useRoute();

// 路由参数
const userId = route.params.id;

// 编程式导航
function goToUser(id) {
  router.push({ name: 'UserProfile', params: { id } });
}

function goBack() {
  router.back();
}

function replaceRoute() {
  router.replace({ name: 'Home' });
}

// 获取路由信息
watch(() => route.params.id, (newId) => {
  // 响应路由变化
  loadUserData(newId);
});
</script>
```

---

## 第三章：Pinia状态管理

### 3.1 Store定义

```javascript
import { defineStore } from 'pinia';
import { ref, computed } from 'vue';

export const useUserStore = defineStore('user', () => {
  // 状态
  const user = ref(null);
  const token = ref(localStorage.getItem('token'));
  const preferences = ref({
    theme: 'light',
    language: 'en'
  });

  // Getters（计算属性）
  const isLoggedIn = computed(() => !!token.value);
  const userName = computed(() => user.value?.name || 'Guest');
  const fullPreferences = computed(() => ({
    ...preferences.value,
    username: userName.value
  }));

  // Actions
  function login(credentials) {
    return api.post('/login', credentials)
      .then(response => {
        token.value = response.token;
        user.value = response.user;
        localStorage.setItem('token', response.token);
      });
  }

  function logout() {
    token.value = null;
    user.value = null;
    localStorage.removeItem('token');
  }

  function updatePreferences(newPrefs) {
    preferences.value = { ...preferences.value, ...newPrefs };
  }

  return {
    user,
    token,
    isLoggedIn,
    userName,
    login,
    logout,
    updatePreferences
  };
});
```

### 3.2 Store使用

```vue
<script setup>
import { storeToRefs } from 'pinia';
import { useUserStore } from './stores/user';

const userStore = useUserStore();

// 解构使用（保持响应式）
const { user, isLoggedIn, userName } = storeToRefs(userStore);

// 方法直接使用
function handleLogin() {
  userStore.login({ username, password });
}

function handleLogout() {
  userStore.logout();
  router.push('/login');
}
</script>
```

---

## 第四章：Vue工程化

### 4.1 Vite构建配置

```javascript
// vite.config.js
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import path from 'path';

export default defineConfig({
  plugins: [vue()],
  
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '~': path.resolve(__dirname, './src/components')
    },
    extensions: ['.vue', '.js', '.jsx', '.ts', '.tsx', '.json']
  },
  
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  },
  
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['vue', 'vue-router', 'pinia'],
          ui: ['element-plus']
        }
      }
    },
    chunkSizeWarningLimit: 1000
  }
});
```

### 4.2 组件测试

```javascript
import { mount } from '@vue/test-utils';
import { describe, it, expect } from 'vitest';
import MyComponent from './MyComponent.vue';

describe('MyComponent', () => {
  it('renders properly', () => {
    const wrapper = mount(MyComponent, {
      props: {
        title: 'Hello World'
      }
    });
    
    expect(wrapper.text()).toContain('Hello World');
  });
  
  it('emits event on click', async () => {
    const wrapper = mount(MyComponent);
    
    await wrapper.find('button').trigger('click');
    
    expect(wrapper.emitted()).toHaveProperty('increment');
  });
  
  it('handles v-model', async () => {
    const wrapper = mount(MyComponent, {
      props: ['modelValue']
    });
    
    await wrapper.find('input').setValue('test');
    
    expect(wrapper.emitted('update:modelValue')).toBeTruthy();
  });
});
```

---

## 参考资源

### 官方文档
- Vue3: vuejs.org
- Vue Router: router.vuejs.org
- Pinia: pinia.vuejs.org
- Vite: vitejs.dev

### 进阶资源
- Vue Mastery
- Vue School
- Anthony Fu（Vue团队）

---

*本知识文件最后更新：2026-02-07*
