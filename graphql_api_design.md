# GraphQL与API设计实战

## 第一章：GraphQL基础

### 1.1 GraphQL核心概念

#### Schema定义
```graphql
# 标量类型
scalar DateTime
scalar Email
scalar UUID

# 枚举类型
enum UserRole {
    ADMIN
    EDITOR
    VIEWER
}

# 接口类型
interface Node {
    id: ID!
}

interface Character {
    id: ID!
    name: String!
    friends: [Character]
}

# 对象类型
type User implements Node {
    id: ID!
    email: Email!
    username: String!
    role: UserRole!
    posts: [Post!]!
    createdAt: DateTime!
}

type Post implements Node {
    id: ID!
    title: String!
    content: String!
    author: User!
    comments: [Comment!]!
    createdAt: DateTime!
}

type Comment {
    id: ID!
    content: String!
    author: User!
    createdAt: DateTime!
}

# 联合类型
union SearchResult = User | Post | Comment

# 查询类型
type Query {
    me: User
    user(id: ID!): User
    users(role: UserRole): [User!]!
    post(id: ID!): Post
    posts(authorId: ID): [Post!]!
    search(term: String!): [SearchResult!]!
}

# 变更类型
type Mutation {
    createUser(input: CreateUserInput!): User!
    updateUser(id: ID!, input: UpdateUserInput!): User!
    deleteUser(id: ID!): Boolean!
    createPost(input: CreatePostInput!): Post!
    addComment(postId: ID!, content: String!): Comment!
}

# 输入类型
input CreateUserInput {
    email: Email!
    username: String!
    password: String!
    role: UserRole = VIEWER
}

input UpdateUserInput {
    email: Email
    username: String
}

input CreatePostInput {
    title: String!
    content: String!
    authorId: ID!
}

# 订阅类型
type Subscription {
    postCreated: Post!
    commentAdded(postId: ID!): Comment!
    userUpdated: User!
}
```

### 1.2 查询与变更

#### 查询示例
```graphql
# 查询用户及其文章
query GetUserWithPosts($userId: ID!) {
    user(id: $userId) {
        id
        username
        email
        posts {
            id
            title
            createdAt
        }
    }
}

# 片段复用
fragment PostFields on Post {
    id
    title
    content
    createdAt
}

query GetPosts($authorId: ID!) {
    posts(authorId: $authorId) {
        ...PostFields
        author {
            username
        }
    }
}

# 变量传递
{
    "userId": "123"
}

# 内联片段
query GetSearchResults($term: String!) {
    search(term: $term) {
        ... on User {
            username
            email
        }
        ... on Post {
            title
            author {
                username
            }
        }
        ... on Comment {
            content
            author {
                username
            }
        }
    }
}
```

#### 变更示例
```graphql
mutation CreateUser($input: CreateUserInput!) {
    createUser(input: $input) {
        id
        email
        username
        role
    }
}

# 多字段变更
mutation CreatePostAndComment {
    createPost(input: { title: "New Post", content: "Content", authorId: "123" }) {
        id
        title
        comments {
            id
            content
        }
    }
}

# 变量输入
{
    "input": {
        "email": "user@example.com",
        "username": "johndoe",
        "password": "securepassword"
    }
}
```

---

## 第二章：Apollo Server实战

### 2.1 Resolver开发

```typescript
import { ApolloServer, gql } from 'apollo-server';
import { User, Post } from './models';

const typeDefs = gql`
    type User {
        id: ID!
        username: String!
        email: String!
        posts: [Post!]!
    }

    type Post {
        id: ID!
        title: String!
        content: String!
        author: User!
        comments: [Comment!]!
    }

    type Comment {
        id: ID!
        content: String!
        author: User!
    }

    type Query {
        user(id: ID!): User
        users: [User!]!
        post(id: ID!): Post
        posts: [Post!]!
    }

    type Mutation {
        createUser(input: CreateUserInput!): User!
        createPost(input: CreatePostInput!): Post!
    }

    input CreateUserInput {
        email: String!
        username: String!
        password: String!
    }

    input CreatePostInput {
        title: String!
        content: String!
        authorId: ID!
    }
`;

const resolvers = {
    Query: {
        user: async (_: any, { id }: { id: string }) => {
            return await User.findById(id);
        },
        
        users: async () => {
            return await User.findAll();
        },
        
        post: async (_: any, { id }: { id: string }) => {
            return await Post.findById(id);
        },
        
        posts: async (_: any, { authorId }: { authorId?: string }) => {
            if (authorId) {
                return await Post.findByAuthor(authorId);
            }
            return await Post.findAll();
        },
    },
    
    Mutation: {
        createUser: async (_: any, { input }: { input: CreateUserInput }) => {
            return await User.create(input);
        },
        
        createPost: async (_: any, { input }: { input: CreatePostInput }) => {
            return await Post.create(input);
        },
    },
    
    // 嵌套Resolver
    User: {
        posts: async (user: User) => {
            return await Post.findByAuthor(user.id);
        },
    },
    
    Post: {
        author: async (post: Post) => {
            return await User.findById(post.authorId);
        },
        
        comments: async (post: Post) => {
            return await Comment.findByPost(post.id);
        },
    },
};

const server = new ApolloServer({
    typeDefs,
    resolvers,
    context: ({ req }) => ({
        user: getUserFromToken(req.headers.authorization),
    }),
});

server.listen().then(({ url }) => {
    console.log(`🚀 Server ready at ${url}`);
});
```

### 2.2 DataLoader批处理

```typescript
import DataLoader from 'dataloader';
import { User, Post } from './models';

// 创建DataLoader
const createLoaders = () => ({
    userLoader: new DataLoader(async (userIds: string[]) => {
        const users = await User.findByIds(userIds);
        return userIds.map(id => users.find(u => u.id === id) || null);
    }),
    
    postLoader: new DataLoader(async (postIds: string[]) => {
        const posts = await Post.findByIds(postIds);
        return postIds.map(id => posts.find(p => p.id === id) || null);
    }),
});

// 在Resolver中使用
const resolvers = {
    Post: {
        author: async (post: Post, _: any, context: Context) => {
            return context.userLoader.load(post.authorId);
        },
    },
    
    User: {
        posts: async (user: User, _: any, context: Context) => {
            const posts = await Post.findByAuthor(user.id);
            return posts.map(p => ({ ...p, authorId: user.id }));
        },
    },
};
```

---

## 第三章：API设计最佳实践

### 3.1 RESTful设计

#### 资源命名规范
```
✅ 正确示例：
GET    /api/users              # 获取用户列表
GET    /api/users/:id          # 获取单个用户
POST   /api/users              # 创建用户
PUT    /api/users/:id          # 更新用户（整体）
PATCH  /api/users/:id          # 部分更新
DELETE /api/users/:id          # 删除用户

# 嵌套资源
GET    /api/users/:id/posts              # 获取用户文章
GET    /api/users/:id/posts/:postId      # 获取特定文章
POST   /api/users/:id/posts              # 为用户创建文章

# 过滤与分页
GET    /api/users?role=admin&page=1&limit=20
GET    /api/posts?authorId=123&status=published
GET    /api/products?category=electronics&price_gte=100

# 搜索
GET    /api/posts/search?q=keyword&sort=created_at&order=desc

# 关系查询
GET    /api/users/:id/followers    # 获取关注者
GET    /api/users/:id/following    # 获取关注的人
GET    /api/posts/:id/comments    # 获取评论
```

### 3.2 API版本控制

```typescript
// 路径版本
GET /api/v1/users
GET /api/v2/users

// Header版本
GET /api/users
Accept: application/vnd.api+json;version=1

// 查询参数版本
GET /api/users?version=1

// Express中实现
app.use('/api/v1', v1Router);
app.use('/api/v2', v2Router);

// GraphQL版本（向后兼容）
type User {
    id: ID!
    username: String!      # v1有，v2保留
    email: String!         # v1有，v2保留
    phone: String         # v2新增
    avatarUrl: String     # v2新增，已废弃@deprecated(reason: "Use profileImage")
}

// 废弃字段标记
type Post {
    id: ID!
    title: String!
    content: String!      # 已废弃
    body: String!         # 新字段
    @deprecated(reason: "Use 'body' field instead")
}
```

---

## 参考资源

### 官方文档
- GraphQL: graphql.org
- Apollo: www.apollographql.com/docs
- RESTful API设计

### 进阶资源
- GraphQL Spec
- API Design Patterns Book
- RESTful Web Services Book

---

*本知识文件最后更新：2026-02-07*
