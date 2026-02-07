# React高级架构与性能优化

## 第一章：React核心概念

### 1.1 Hooks深入理解

#### useState与useReducer
```jsx
// useState：简单状态
function Counter() {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(c => c + 1)}>{count}</button>;
}

// useReducer：复杂状态
const initialState = { count: 0, step: 1 };

function reducer(state, action) {
  switch (action.type) {
    case 'increment':
      return { ...state, count: state.count + state.step };
    case 'decrement':
      return { ...state, count: state.count - state.step };
    case 'setStep':
      return { ...state, step: action.payload };
    default:
      return state;
  }
}

function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <>
      <p>Count: {state.count}</p>
      <button onClick={() => dispatch({ type: 'increment' })}>+</button>
      <button onClick={() => dispatch({ type: 'decrement' })}>-</button>
    </>
  );
}
```

#### useEffect依赖管理
```jsx
// 正确的依赖数组
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);

  // ✅ 正确：包含所有依赖
  useEffect(() => {
    const fetchUser = async () => {
      const data = await api.getUser(userId);
      setUser(data);
    };
    fetchUser();
  }, [userId]); // ✅ 只依赖userId

  // ❌ 错误：依赖函数会变化
  useEffect(() => {
    document.title = user ? `${user.name}'s Profile` : 'Loading';
  }, [user]); // ✅ 正确：依赖user对象

  // 清理函数
  useEffect(() => {
    const subscription = subscribe(userId);
    return () => subscription.unsubscribe(); // 清理
  }, [userId]);
}
```

### 1.2 自定义Hooks

```jsx
// useDebounce：防抖Hook
function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => clearTimeout(handler);
  }, [value, delay]);

  return debouncedValue;
}

// useLocalStorage：本地存储Hook
function useLocalStorage(key, initialValue) {
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      return initialValue;
    }
  });

  const setValue = (value) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(error);
    }
  };

  return [storedValue, setValue];
}
```

---

## 第二章：React性能优化

### 2.1 React.memo与useMemo

```jsx
// React.memo：组件记忆化
const ExpensiveComponent = React.memo(function ExpensiveComponent({ data }) {
  // 只在data变化时重新渲染
  return <div>{/* expensive rendering */}</div>;
}, (prevProps, nextProps) => {
  // 自定义比较函数
  return prevProps.data.id === nextProps.data.id;
});

// useMemo：值记忆化
function ProductList({ products, filter }) {
  const filteredProducts = useMemo(() => {
    return products.filter(p => p.category === filter);
  }, [products, filter]); // ✅ 只在依赖变化时重新计算

  return (
    <ul>
      {filteredProducts.map(p => <li key={p.id}>{p.name}</li>)}
    </ul>
  );
}

// useCallback：函数记忆化
function Parent() {
  const [count, setCount] = useState(0);

  // ✅ 回调函数被记忆，避免子组件不必要的重渲染
  const handleClick = useCallback(() => {
    console.log('Clicked:', count);
  }, [count]);

  return <Child onClick={handleClick} />;
}
```

### 2.2 代码分割与懒加载

```jsx
import React, { Suspense, lazy } from 'react';

// 路由级别代码分割
const Dashboard = lazy(() => import('./Dashboard'));
const Settings = lazy(() => import('./Settings'));
const Analytics = lazy(() => import('./Analytics'));

function App() {
  return (
    <Router>
      <Suspense fallback={<LoadingSpinner />}>
        <Routes>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/analytics" element={<Analytics />} />
        </Routes>
      </Suspense>
    </Router>
  );
}

// 组件级别懒加载
function ProductPage({ productId }) {
  const ProductDetails = lazy(() => import('./ProductDetails'));
  const ProductReviews = lazy(() => import('./ProductReviews'));

  return (
    <div>
      <h1>{productId}</h1>
      <Suspense fallback={<div>Loading details...</div>}>
        <ProductDetails productId={productId} />
      </Suspense>
      <Suspense fallback={<div>Loading reviews...</div>}>
        <ProductReviews productId={productId} />
      </Suspense>
    </div>
  );
}
```

---

## 第三章：状态管理

### 3.1 Zustand状态管理

```jsx
import { create } from 'zustand';

// 创建store
const useStore = create((set, get) => ({
  // 状态
  user: null,
  cart: [],
  products: [],
  
  // actions
  setUser: (user) => set({ user }),
  
  addToCart: (product) => set((state) => ({
    cart: [...state.cart, product]
  })),
  
  removeFromCart: (productId) => set((state) => ({
    cart: state.cart.filter(item => item.id !== productId)
  })),
  
  // 异步action
  fetchProducts: async (category) => {
    const data = await api.getProducts(category);
    set({ products: data });
  },
  
  // 计算属性
  cartTotal: () => get().cart.reduce((sum, item) => sum + item.price, 0),
}));

// 在组件中使用
function CartButton() {
  const cart = useStore((state) => state.cart);
  const cartTotal = useStore((state) => state.cartTotal);

  return (
    <button>
      Cart ({cart.length}) - ${cartTotal()}
    </button>
  );
}
```

### 3.2 Redux Toolkit

```jsx
import { createSlice, configureStore } from '@reduxjs/toolkit';

// 创建slice
const counterSlice = createSlice({
  name: 'counter',
  initialState: { value: 0 },
  reducers: {
    increment: (state) => { state.value += 1; },
    decrement: (state) => { state.value -= 1; },
    incrementByAmount: (state, action) => {
      state.value += action.payload;
    },
  },
});

export const { increment, decrement, incrementByAmount } = counterSlice.actions;

// 创建store
const store = configureStore({
  reducer: {
    counter: counterSlice.reducer,
  },
});

// 异步Thunk
export const fetchUserData = (userId) => async (dispatch) => {
  dispatch(setLoading(true));
  try {
    const user = await api.getUser(userId);
    dispatch(setUser(user));
  } catch (error) {
    dispatch(setError(error.message));
  } finally {
    dispatch(setLoading(false));
  }
};
```

---

## 第四章：React测试

### 4.1 Jest测试

```jsx
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// 单元测试
test('counter increments when button is clicked', () => {
  render(<Counter />);
  
  const button = screen.getByRole('button', { name: /increment/i });
  const display = screen.getByText(/count: 0/i);
  
  fireEvent.click(button);
  expect(display).toHaveTextContent(/count: 1/i);
});

// 用户事件测试
test('form submission', async () => {
  const user = userEvent.setup();
  render(<LoginForm />);
  
  const emailInput = screen.getByLabelText(/email/i);
  const submitButton = screen.getByRole('button', { name: /submit/i });
  
  await user.type(emailInput, 'test@example.com');
  await user.click(submitButton);
  
  expect(screen.getByText(/success/i)).toBeInTheDocument();
});

// 异步组件测试
test('loads and displays user data', async () => {
  render(<UserProfile userId={123} />);
  
  expect(screen.getByText(/loading/i)).toBeInTheDocument();
  
  await screen.findByText(/john doe/i);
  expect(screen.queryByText(/loading/i)).not.toBeInTheDocument();
});
```

---

## 参考资源

### 官方文档
- React: react.dev
- Redux: redux.js.org
- Zustand: github.com/pmndrs/zustand

### 进阶学习
- React Conf Videos
- Kent C. Dodds Blog
- Epic React Course

---

*本知识文件最后更新：2026-02-07*
