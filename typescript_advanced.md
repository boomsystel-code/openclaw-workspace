# TypeScript高级编程

## 第一章：TypeScript核心深入

### 1.1 类型系统进阶

#### 高级类型
```typescript
// 交叉类型（Intersection Types）
type Admin = { adminLevel: number };
type Employee = { employeeId: string };
type AdminEmployee = Admin & Employee;

// 联合类型（Union Types）
type Status = 'pending' | 'in_progress' | 'completed';

// 类型守卫
function isString(value: unknown): value is string {
    return typeof value === 'string';
}

function processValue(value: string | number) {
    if (isString(value)) {
        // value在if块中是string类型
        console.log(value.toUpperCase());
    } else {
        // value在else块中是number类型
        console.log(value.toFixed(2));
    }
}

// 可辨识联合（Discriminated Unions）
type Shape = 
    | { kind: 'circle'; radius: number }
    | { kind: 'rectangle'; width: number; height: number }
    | { kind: 'triangle'; base: number; height: number };

function getArea(shape: Shape): number {
    switch (shape.kind) {
        case 'circle':
            return Math.PI * shape.radius ** 2;
        case 'rectangle':
            return shape.width * shape.height;
        case 'triangle':
            return 0.5 * shape.base * shape.height;
    }
}
```

#### 泛型进阶
```typescript
// 泛型约束
interface HasLength {
    length: number;
}

function logLength<T extends HasLength>(arg: T): T {
    console.log(arg.length);
    return arg;
}

// 泛型默认值
interface ApiResponse<T = any> {
    data: T;
    status: number;
    message: string;
}

// 泛型条件类型
type IsString<T> = T extends string ? true : false;
type A = IsString<string>;  // true
type B = IsString<number>; // false

// 映射类型
type Readonly<T> = {
    readonly [P in keyof T]: T[P];
};

type Partial<T> = {
    [P in keyof T]?: T[P];
};

// 内置工具类型实现
type Required<T> = {
    [P in keyof T]-?: T[P];
};

type Record<K extends keyof any, T> = {
    [P in K]: T;
};
```

### 1.2 装饰器与元编程

```typescript
// 类装饰器
function logClass(target: Function) {
    console.log(`Class ${target.name} was defined`);
}

@logClass
class MyClass {
    // 类定义
}

// 方法装饰器
function logMethod(
    target: Object,
    propertyKey: string,
    descriptor: PropertyDescriptor
) {
    const originalMethod = descriptor.value;
    
    descriptor.value = function(...args: unknown[]) {
        console.log(`Method ${propertyKey} called with:`, args);
        const result = originalMethod.apply(this, args);
        console.log(`Method ${propertyKey} returned:`, result);
        return result;
    };
}

class Calculator {
    @logMethod
    add(a: number, b: number): number {
        return a + b;
    }
}

// 参数装饰器
function logParameter(
    target: Object,
    propertyKey: string,
    parameterIndex: number
) {
    console.log(`Parameter at index ${parameterIndex} of ${propertyKey}`);
}
```

---

## 第二章：TypeScript与框架

### 2.1 React + TypeScript

```typescript
// 组件类型定义
interface ButtonProps {
    readonly children: React.ReactNode;
    readonly onClick?: () => void;
    readonly variant?: 'primary' | 'secondary' | 'danger';
    readonly disabled?: boolean;
}

const Button: React.FC<ButtonProps> = ({
    children,
    onClick,
    variant = 'primary',
    disabled = false,
}) => {
    const className = `btn btn-${variant}`;
    
    return (
        <button
            className={className}
            onClick={onClick}
            disabled={disabled}
        >
            {children}
        </button>
    );
};

// Hook类型定义
function useLocalStorage<T>(
    key: string,
    initialValue: T
): [T, (value: T | ((val: T) => T)) => void] {
    const [storedValue, setStoredValue] = useState<T>(() => {
        try {
            const item = window.localStorage.getItem(key);
            return item ? JSON.parse(item) : initialValue;
        } catch {
            return initialValue;
        }
    });
    
    const setValue = (value: T | ((val: T) => T)) => {
        try {
            const valueToStore = value instanceof Function 
                ? value(storedValue) 
                : value;
            setStoredValue(valueToStore);
            window.localStorage.setItem(key, JSON.stringify(valueToStore));
        } catch (error) {
            console.error(error);
        }
    };
    
    return [storedValue, setValue];
}

// Context类型定义
interface User {
    id: string;
    name: string;
    email: string;
}

interface UserContextType {
    user: User | null;
    login: (user: User) => void;
    logout: () => void;
}

const UserContext = React.createContext<UserContextType | null>(null);
```

### 2.2 Node.js + TypeScript

```typescript
// Express类型扩展
declare module 'express' {
    interface Request {
        user?: JwtPayload;
        traceId?: string;
    }
}

// 中间件类型
interface AsyncRequestHandler {
    (
        req: Request,
        res: Response,
        next: NextFunction
    ): Promise<void>;
}

function asyncHandler(
    fn: AsyncRequestHandler
): RequestHandler {
    return (req, res, next) => {
        Promise.resolve(fn(req, res, next)).catch(next);
    };
}

// 路由类型定义
interface RouteDefinition {
    path: string;
    method: 'get' | 'post' | 'put' | 'delete' | 'patch';
    handler: RequestHandler;
    middlewares?: RequestHandler[];
}

class Router {
    private routes: RouteDefinition[] = [];
    
    addRoute(route: RouteDefinition): this {
        this.routes.push(route);
        return this;
    }
    
    get(path: string, handler: RequestHandler): this {
        return this.addRoute({ path, method: 'get', handler });
    }
}
```

---

## 第三章：设计模式与工程实践

### 3.1 依赖注入

```typescript
// 简单DI容器
class Container {
    private dependencies = new Map<string, any>();
    private singletons = new Map<string, any>();
    
    register<T>(
        token: string,
        factory: () => T,
        singleton = true
    ): void {
        this.dependencies.set(token, { factory, singleton });
    }
    
    resolve<T>(token: string): T {
        const dep = this.dependencies.get(token);
        if (!dep) {
            throw new Error(`Dependency ${token} not found`);
        }
        
        if (dep.singleton) {
            if (!this.singletons.has(token)) {
                this.singletons.set(token, dep.factory());
            }
            return this.singletons.get(token);
        }
        
        return dep.factory();
    }
}

// 使用装饰器的依赖注入
function Injectable(token?: string) {
    return function<T extends Function>(constructor: T) {
        container.register(
            token || constructor.name,
            () => new constructor()
        );
    };
}

@Injectable('Logger')
class LoggerService {
    log(message: string): void {
        console.log(`[LOG] ${message}`);
    }
}

@Injectable('UserService')
class UserService {
    constructor(@Inject('Logger') private logger: LoggerService) {}
    
    getUser(id: string): User {
        this.logger.log(`Getting user ${id}`);
        return { id, name: 'User' };
    }
}
```

### 3.2 错误处理

```typescript
// 自定义错误类
class AppError extends Error {
    constructor(
        message: string,
        public statusCode: number,
        public code?: string
    ) {
        super(message);
        Error.captureStackTrace(this, this.constructor);
    }
}

// 错误处理中间件
function errorHandler(
    err: Error,
    req: Request,
    res: Response,
    next: NextFunction
): void {
    if (err instanceof AppError) {
        res.status(err.statusCode).json({
            success: false,
            error: {
                message: err.message,
                code: err.code
            }
        });
        return;
    }
    
    // 未知错误
    console.error('Unknown error:', err);
    res.status(500).json({
        success: false,
        error: {
            message: 'Internal server error',
            code: 'INTERNAL_ERROR'
        }
    });
}

// Result模式
type Result<T, E = Error> = 
    | { ok: true; value: T }
    | { ok: false; error: E };

function safeParse<T>(json: string): Result<T, SyntaxError> {
    try {
        return { ok: true, value: JSON.parse(json) };
    } catch (error) {
        return { ok: false, error: error as SyntaxError };
    }
}
```

---

## 参考资源

### 官方文档
- TypeScript: www.typescriptlang.org/docs
- React TypeScript CheatSheet
- TypeScript Deep Dive

### 进阶资源
- Advanced TypeScript
- TypeScript Team Blog
- Effective TypeScript Book

---

*本知识文件最后更新：2026-02-07*
